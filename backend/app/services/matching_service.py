import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.candidate import Candidate
from app.models.job import Job
from app.schemas.candidate import CandidateResponse, RankJobResponse
from app.services.text_preprocessor import TextPreprocessor
from app.services.explainability_service import ExplainabilityService
from app.services.resume_parser import ResumeParser

logger = logging.getLogger(__name__)

def utc_now():
    return datetime.now(timezone.utc)

LABEL_NAMES = ["No Fit", "Potential Fit", "Good Fit"]

class PairwiseMLP(nn.Module):
    def __init__(self, input_dim: int = 1536, hidden_dim1: int = 256, hidden_dim2: int = 64, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class FineTunedMatchingHead(nn.Module):
    def __init__(self, input_dim: int = 1536, hidden_dim1: int = 256, hidden_dim2: int = 64, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LayerNorm(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MatchingService:
    _encoder = None
    _matching_head = None
    _model_manifest = None
    _model_version = "Verisift-FineTuned-SBERT-v2.0"
    _is_finetuned = False

    @classmethod
    def _load_model(cls):
        """
        Lazy-loads the fine-tuned SBERT encoder and trained matching head.
        Falls back seamlessly to pretrained MiniLM + E4 MLP if fine-tuned checkpoint is missing.
        """
        if cls._matching_head is not None and cls._encoder is not None:
            return

        from sentence_transformers import SentenceTransformer

        project_root = Path(__file__).resolve().parent.parent.parent.parent
        
        # Primary: Fine-tuned SBERT model
        ft_dir = project_root / "artifacts" / "finetuned_sbert"
        ft_encoder_dir = ft_dir / "sbert_encoder"
        ft_head_path = ft_dir / "matching_head.pt"
        ft_manifest_path = ft_dir / "model_manifest.json"

        # Fallback: Baseline MLP
        base_dir = project_root / "artifacts" / "semantic_ranker"
        base_head_path = base_dir / "semantic_ranker_mlp.pt"
        base_manifest_path = base_dir / "model_manifest.json"

        if ft_encoder_dir.exists() and ft_head_path.exists():
            logger.info(f"Loading Fine-Tuned SBERT encoder from: {ft_encoder_dir}")
            cls._encoder = SentenceTransformer(str(ft_encoder_dir), device="cpu")
            
            logger.info(f"Loading Fine-Tuned matching head from: {ft_head_path}")
            head = FineTunedMatchingHead()
            state_dict = torch.load(ft_head_path, map_location="cpu")
            head.net.load_state_dict(state_dict)
            head.eval()
            cls._matching_head = head
            cls._model_version = "Verisift-FineTuned-SBERT-v2.0 (sentence-transformers/all-MiniLM-L6-v2 + Evidence Head)"
            cls._is_finetuned = True
            
            if ft_manifest_path.exists():
                try:
                    with open(ft_manifest_path, "r", encoding="utf-8") as f:
                        cls._model_manifest = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load fine-tuned manifest: {e}")
        elif base_head_path.exists():
            logger.info("Loading baseline SentenceTransformer encoder: sentence-transformers/all-MiniLM-L6-v2")
            cls._encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
            
            logger.info(f"Loading baseline PyTorch MLP matching head from: {base_head_path}")
            mlp = PairwiseMLP()
            state_dict = torch.load(base_head_path, map_location="cpu")
            mlp.load_state_dict(state_dict)
            mlp.eval()
            cls._matching_head = mlp
            cls._model_version = "E4 PyTorch Pairwise MLP (all-MiniLM-L6-v2)"
            cls._is_finetuned = False

            if base_manifest_path.exists():
                try:
                    with open(base_manifest_path, "r", encoding="utf-8") as f:
                        cls._model_manifest = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load baseline manifest: {e}")
        else:
            raise FileNotFoundError(
                f"No candidate matching model weights found at either {ft_head_path} or {base_head_path}."
            )

        logger.info(f"MatchingService initialized successfully with {cls._model_version}.")

    @classmethod
    def compute_pairwise_features(cls, resume_emb: np.ndarray, jd_emb: np.ndarray) -> np.ndarray:
        """Constructs rich pair feature vector [r, j, |r-j|, r*j] (dimension 1536)."""
        diff = np.abs(resume_emb - jd_emb)
        prod = resume_emb * jd_emb
        return np.concatenate([resume_emb, jd_emb, diff, prod], axis=-1)

    @classmethod
    def encode_section_aware_resume(cls, resume_text: str, structured_resume: Optional[Dict[str, Any]]) -> np.ndarray:
        """
        Extracts and encodes multiple meaningful resume sections (header/summary, skills, experience, projects)
        and mean-pools their embeddings. Prevents late-document truncation from discarding technical skills or internships.
        """
        if not resume_text.strip():
            # Return zero unit vector
            return np.zeros(384, dtype=np.float32)

        chunks: List[str] = [resume_text[:400]]  # Lead section (summary, education)

        if structured_resume and "sections" in structured_resume:
            sec = structured_resume["sections"]
            if "skills" in sec and sec["skills"]:
                skills_val = "Technical Skills: " + (" ".join(sec["skills"]) if isinstance(sec["skills"], list) else str(sec["skills"]))
                chunks.append(skills_val[:500])
            if "experience" in sec and sec["experience"]:
                exp_val = "Work Experience: " + (" ".join(sec["experience"]) if isinstance(sec["experience"], list) else str(sec["experience"]))
                chunks.append(exp_val[:600])
            if "projects" in sec and sec["projects"]:
                proj_val = "Engineering Projects: " + (" ".join(sec["projects"]) if isinstance(sec["projects"], list) else str(sec["projects"]))
                chunks.append(proj_val[:500])
        elif len(resume_text) > 400:
            # Fallback sliding window chunks without structured boundaries
            chunks.append(resume_text[400:1000])
            if len(resume_text) > 1000:
                chunks.append(resume_text[1000:1800])

        embs = cls._encoder.encode(chunks, normalize_embeddings=True)
        pooled = np.mean(embs, axis=0)
        norm = np.linalg.norm(pooled)
        if norm > 1e-9:
            pooled = pooled / norm
        return pooled

    @classmethod
    def calculate_relevance_score(cls, probabilities: np.ndarray) -> float:
        """
        Continuous ordinal relevance score out of 100:
        Score = 100 * (0.0 * P(No Fit) + 0.5 * P(Potential Fit) + 1.0 * P(Good Fit))
        """
        score = 100.0 * (0.0 * probabilities[0] + 0.5 * probabilities[1] + 1.0 * probabilities[2])
        return float(np.clip(score, 0.0, 100.0))

    @classmethod
    def rank_job_candidates(
        cls,
        db: Session,
        job_id: int,
        recruiter_id: int
    ) -> RankJobResponse:
        """
        Scores and ranks all candidates for a job role with verifiable explainability.
        1. Validates job ownership.
        2. Retrieves candidates with completed text extraction.
        3. Encodes job description and section-aware resumes.
        4. Runs pairwise neural inference + cosine semantic similarity.
        5. Evaluates required and preferred competencies against verified resume text.
        6. Synthesizes transparent score breakdown and peer comparison explanations.
        7. Sorts candidates by deterministic tie-breaking (relevance score, required skill coverage, experience score, ID).
        8. Persists scores, ranks, matched skills, and full explanation JSON.
        """
        try:
            cls._load_model()
        except FileNotFoundError as e:
            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Matching engine model weights unavailable: {e}"
            )
        except Exception as e:
            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize matching engine: {e}"
            )

        job = db.query(Job).filter(Job.id == job_id, Job.recruiter_id == recruiter_id).first()
        if not job:
            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job role not found or access denied."
            )

        candidates = db.query(Candidate).filter(
            Candidate.job_id == job_id,
            Candidate.recruiter_id == recruiter_id,
            Candidate.processing_status == "completed"
        ).all()

        if not candidates:
            return RankJobResponse(
                job_id=job.id,
                job_title=job.title,
                model_used=cls._model_version,
                total_candidates_ranked=0,
                ranked_candidates=[],
                ranked_at=utc_now()
            )

        # Preprocess and encode job text (title + description)
        job_full_text = f"{job.title} {job.description}" if job.description else job.title
        jd_clean = TextPreprocessor.normalize(job_full_text)
        jd_emb = cls._encoder.encode([jd_clean], normalize_embeddings=True)[0]

        candidate_items = []
        for cand in candidates:
            resume_text = cand.extracted_text or ""
            is_parser_error = (cand.processing_status == "failed" or not resume_text.strip())

            # Load or build structured resume
            struct_dict = None
            if cand.structured_resume:
                try:
                    struct_dict = json.loads(cand.structured_resume)
                except Exception:
                    struct_dict = None
            if struct_dict is None and resume_text.strip():
                try:
                    struct_dict = ResumeParser.parse_resume(resume_text).to_dict()
                    cand.structured_resume = json.dumps(struct_dict)
                except Exception:
                    struct_dict = None

            # Section-aware pooled candidate embedding
            r_emb = cls.encode_section_aware_resume(resume_text, struct_dict)

            # Pairwise neural inference
            pair_feat = cls.compute_pairwise_features(r_emb, jd_emb)
            x_tensor = torch.tensor(pair_feat, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                logits = cls._matching_head(x_tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

            probs = np.nan_to_num(probs, nan=0.0)
            if probs.sum() > 0:
                probs = probs / probs.sum()
            else:
                probs = np.array([1.0, 0.0, 0.0])

            # Continuous semantic score combining neural classification & cosine alignment
            neural_score = cls.calculate_relevance_score(probs)
            cos_sim = float(np.dot(r_emb, jd_emb))
            cos_score = float(np.clip((cos_sim + 1.0) / 2.0 * 100.0, 0.0, 100.0))
            
            # Weighted semantic score (70% neural MLP + 30% direct cosine similarity)
            semantic_score = round(0.70 * neural_score + 0.30 * cos_score, 2)

            # Generate evidence-grounded explainability assessment
            cand_name = cand.full_name or "Candidate"
            explanation_obj = ExplainabilityService.generate_candidate_explanation(
                candidate_name=cand_name,
                job_title=job.title,
                job_description=job.description or "",
                resume_text=resume_text,
                semantic_score=semantic_score,
                structured_resume=struct_dict,
                is_parser_error=is_parser_error
            )

            candidate_items.append({
                "candidate": cand,
                "explanation_obj": explanation_obj,
                "probabilities": {
                    "no_fit": round(float(probs[0]), 4),
                    "potential_fit": round(float(probs[1]), 4),
                    "good_fit": round(float(probs[2]), 4)
                }
            })

        # Deterministic multi-candidate tie-breaking:
        # 1. Final reconciled relevance score (DESC)
        # 2. Required skill coverage percentage (DESC)
        # 3. Practical experience & internship score (DESC)
        # 4. Candidate ID (ASC)
        candidate_items.sort(
            key=lambda item: (
                item["explanation_obj"].relevance_score,
                item["explanation_obj"].score_breakdown["required_skill_coverage"],
                item["explanation_obj"].score_breakdown["experience_project_score"],
                -item["candidate"].id
            ),
            reverse=True
        )

        # Assign evidence-grounded peer comparisons
        explanations_list = [item["explanation_obj"] for item in candidate_items]
        ExplainabilityService.assign_peer_comparisons(explanations_list)

        # Update and persist database records
        ranked_responses: List[CandidateResponse] = []
        for rank_idx, item in enumerate(candidate_items, start=1):
            cand = item["candidate"]
            expl = item["explanation_obj"]

            cand.relevance_score = expl.relevance_score
            cand.fit_category = expl.fit_category
            cand.ranking_position = rank_idx
            cand.predicted_probabilities = json.dumps(item["probabilities"])
            cand.model_version = cls._model_version
            cand.updated_at = utc_now()

            # Matched skills string for concise indexing
            matched_skills_list = [
                s["skill_name"] for s in expl.required_skills if s["status"] in ("Matched", "Partial")
            ] + [
                s["skill_name"] for s in expl.preferred_skills if s["status"] in ("Matched", "Partial")
            ]
            cand.matched_skills = json.dumps(list(dict.fromkeys(matched_skills_list)))
            cand.explanation = json.dumps(asdict(expl))

            ranked_responses.append(CandidateResponse(
                id=cand.id,
                job_id=cand.job_id,
                recruiter_id=cand.recruiter_id,
                original_filename=cand.original_filename,
                file_type=cand.file_type,
                file_size=cand.file_size,
                extracted_text=cand.extracted_text,
                processing_status=cand.processing_status,
                error_message=cand.error_message,
                relevance_score=cand.relevance_score,
                fit_category=cand.fit_category,
                ranking_position=cand.ranking_position,
                predicted_probabilities=cand.predicted_probabilities,
                model_version=cand.model_version,
                created_at=cand.created_at,
                updated_at=cand.updated_at,
                job_title=job.title,
                full_name=cand.full_name,
                email=cand.email,
                phone=cand.phone,
                location=cand.location,
                linkedin_url=cand.linkedin_url,
                github_url=cand.github_url,
                portfolio_url=cand.portfolio_url,
                matched_skills=cand.matched_skills,
                structured_resume=cand.structured_resume,
                explanation=cand.explanation
            ))

        db.commit()

        return RankJobResponse(
            job_id=job.id,
            job_title=job.title,
            model_used=cls._model_version,
            total_candidates_ranked=len(ranked_responses),
            ranked_candidates=ranked_responses,
            ranked_at=utc_now()
        )
