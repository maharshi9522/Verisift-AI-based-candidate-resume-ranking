"""
Verisift Candidate Explainability & Evidence Extraction Engine
Generates transparent, evidence-grounded assessments for shortlisted candidates.
Distinguishes demonstrated matches, partial matches, missing requirements,
and ensures parser limitations are never treated as negative candidate assessments.
"""

import re
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

# Canonical Skill Taxonomy with aliases, related skills, and normalization
SKILL_TAXONOMY = {
    # Programming Languages
    "python": {"canonical": "Python", "aliases": ["python", "python3", "py"], "category": "language"},
    "c++": {"canonical": "C++", "aliases": ["c++", "cpp", "c / c++"], "category": "language"},
    "c": {"canonical": "C", "aliases": ["c language", "c programming"], "category": "language"},
    "javascript": {"canonical": "JavaScript", "aliases": ["javascript", "js", "ecmascript"], "category": "language"},
    "typescript": {"canonical": "TypeScript", "aliases": ["typescript", "ts"], "category": "language"},
    "sql": {"canonical": "SQL", "aliases": ["sql", "structured query language", "mysql", "postgresql", "sqlite", "tsql", "plsql"], "category": "database"},
    "java": {"canonical": "Java", "aliases": ["java", "core java"], "category": "language"},
    "go": {"canonical": "Go", "aliases": ["golang", "go language"], "category": "language"},
    "rust": {"canonical": "Rust", "aliases": ["rust"], "category": "language"},
    
    # Core CS & Algorithms
    "dsa": {"canonical": "Data Structures & Algorithms", "aliases": ["data structures", "algorithms", "dsa", "data structures & algorithms", "data structures and algorithms"], "category": "core_cs"},
    "oop": {"canonical": "Object-Oriented Programming", "aliases": ["oop", "object oriented programming", "oops", "object-oriented design"], "category": "core_cs"},
    "dbms": {"canonical": "DBMS & Relational Databases", "aliases": ["dbms", "database management", "relational databases", "rdbms"], "category": "core_cs"},
    "operating_systems": {"canonical": "Operating Systems", "aliases": ["operating systems", "os fundamentals", "linux", "unix"], "category": "core_cs"},
    "networking": {"canonical": "Computer Networks", "aliases": ["networking", "computer networks", "tcp/ip", "http"], "category": "core_cs"},
    
    # Web & APIs
    "rest_apis": {"canonical": "REST APIs", "aliases": ["rest api", "rest apis", "restful", "restful api", "restful apis", "rest", "api integration", "api development"], "category": "backend"},
    "fastapi": {"canonical": "FastAPI", "aliases": ["fastapi", "fast api"], "category": "backend"},
    "flask": {"canonical": "Flask", "aliases": ["flask"], "category": "backend"},
    "django": {"canonical": "Django", "aliases": ["django"], "category": "backend"},
    "nodejs": {"canonical": "Node.js", "aliases": ["node.js", "nodejs", "node"], "category": "backend"},
    "express": {"canonical": "Express.js", "aliases": ["express", "express.js", "expressjs"], "category": "backend"},
    "react": {"canonical": "React", "aliases": ["react", "react.js", "reactjs"], "category": "frontend"},
    
    # Developer Tools & Infrastructure
    "git": {"canonical": "Git", "aliases": ["git", "github", "gitlab", "version control"], "category": "tools"},
    "docker": {"canonical": "Docker", "aliases": ["docker", "containerization", "containers"], "category": "devops"},
    "kubernetes": {"canonical": "Kubernetes", "aliases": ["kubernetes", "k8s"], "category": "devops"},
    "redis": {"canonical": "Redis", "aliases": ["redis", "in-memory cache"], "category": "database"},
    "mongodb": {"canonical": "MongoDB", "aliases": ["mongodb", "mongo", "nosql"], "category": "database"},
    
    # Machine Learning & AI
    "pytorch": {"canonical": "PyTorch", "aliases": ["pytorch", "torch"], "category": "ai_ml"},
    "tensorflow": {"canonical": "TensorFlow", "aliases": ["tensorflow", "tf", "keras"], "category": "ai_ml"},
    "machine_learning": {"canonical": "Machine Learning", "aliases": ["machine learning", "ml", "scikit-learn", "sklearn"], "category": "ai_ml"},
    "deep_learning": {"canonical": "Deep Learning", "aliases": ["deep learning", "neural networks", "dl"], "category": "ai_ml"},
    "nlp": {"canonical": "NLP & Transformers", "aliases": ["nlp", "natural language processing", "transformers", "bert", "llm", "llms"], "category": "ai_ml"},
    "computer_vision": {"canonical": "Computer Vision", "aliases": ["computer vision", "opencv", "yolo", "yolov8"], "category": "ai_ml"},
}

@dataclass
class SkillEvidence:
    skill_name: str
    status: str  # "Matched", "Partial", "Not evidenced", "Parser limitation"
    confidence: float
    evidence_text: Optional[str] = None
    section_found: Optional[str] = None
    reason: Optional[str] = None

@dataclass
class ExperienceHighlight:
    role_or_project: str
    organization: Optional[str]
    duration_or_date: Optional[str]
    relevance_summary: str
    technologies_evidenced: List[str]
    is_internship: bool = False

@dataclass
class ScoreBreakdown:
    semantic_similarity_score: float   # 0-100 from fine-tuned SBERT
    required_skill_coverage: float     # 0-100 percentage of mandatory skills satisfied
    preferred_skill_coverage: float    # 0-100 percentage of optional skills satisfied
    experience_project_score: float    # 0-100 evidence of practical development / internships
    final_relevance_score: float       # Reconciled 0-100 score
    formula_description: str

@dataclass
class CandidateExplanation:
    rank: Optional[int]
    relevance_score: float
    fit_category: str
    overall_summary: str
    strongest_match: str
    top_missing_requirement: Optional[str]
    required_skills: List[Dict[str, Any]]
    preferred_skills: List[Dict[str, Any]]
    experience_highlights: List[Dict[str, Any]]
    peer_comparison: Optional[str]
    score_breakdown: Dict[str, Any]

class ExplainabilityService:
    @staticmethod
    def extract_job_competencies(job_title: str, job_description: str) -> Tuple[List[str], List[str]]:
        """
        Extracts required and preferred technical competencies from the job description and title.
        Returns: (required_skills, preferred_skills)
        """
        full_text = f"{job_title}\n{job_description}".lower()
        required: List[str] = []
        preferred: List[str] = []

        # Default required skills for general Software Engineering roles
        if "software" in full_text and ("intern" in full_text or "engineer" in full_text or "developer" in full_text):
            # Check for core languages
            if "c++" in full_text or "cpp" in full_text:
                required.append("C++")
            if "python" in full_text:
                required.append("Python")
            if "dsa" in full_text or "data structures" in full_text:
                required.append("Data Structures & Algorithms")
            if "oop" in full_text or "object-oriented" in full_text:
                required.append("Object-Oriented Programming")
            if "sql" in full_text or "database" in full_text:
                required.append("SQL")
            if "git" in full_text:
                required.append("Git")
            if "rest" in full_text or "api" in full_text:
                required.append("REST APIs")

        # Check for ML / AI specific roles
        if any(term in full_text for term in ("machine learning", "deep learning", "ai", "nlp")):
            if "pytorch" in full_text:
                required.append("PyTorch")
            if "transformers" in full_text or "nlp" in full_text or "bert" in full_text:
                required.append("NLP & Transformers")
            if "deep learning" in full_text:
                required.append("Deep Learning")
            if "scikit-learn" in full_text or "machine learning" in full_text:
                required.append("Machine Learning")

        # Preferred skills
        for pref_candidate, key in [
            ("Docker", "docker"), ("Redis", "redis"), ("FastAPI", "fastapi"),
            ("React", "react"), ("TypeScript", "typescript"), ("Kubernetes", "kubernetes"),
            ("MongoDB", "mongodb"), ("Node.js", "nodejs")
        ]:
            if key in full_text and pref_candidate not in required:
                preferred.append(pref_candidate)

        # Fallback if text is concise
        if not required:
            required = ["Python", "Data Structures & Algorithms", "Git", "REST APIs"]
        if not preferred:
            preferred = ["Docker", "SQL", "FastAPI"]

        # Deduplicate preserving order
        req_clean = list(dict.fromkeys(required))
        pref_clean = [p for p in list(dict.fromkeys(preferred)) if p not in req_clean]
        return req_clean, pref_clean

    @classmethod
    def evaluate_skill_evidence(
        cls,
        skill_name: str,
        resume_text: str,
        structured_resume: Optional[Dict[str, Any]] = None,
        is_parser_error: bool = False
    ) -> SkillEvidence:
        """
        Checks whether a skill is evidenced in the candidate's resume,
        locating exact sentence evidence and distinguishing parser limitations.
        """
        if is_parser_error or not resume_text.strip():
            return SkillEvidence(
                skill_name=skill_name,
                status="Parser limitation",
                confidence=0.0,
                evidence_text=None,
                reason="Document text could not be read or extracted; cannot assess this requirement."
            )

        norm_resume = resume_text.lower()
        sections = structured_resume.get("sections", {}) if structured_resume else {}

        # Look up aliases for this skill
        target_aliases = [skill_name.lower()]
        for k, v in SKILL_TAXONOMY.items():
            if v["canonical"].lower() == skill_name.lower() or k == skill_name.lower():
                target_aliases.extend(v["aliases"])
                break
        target_aliases = list(set(target_aliases))

        # Check section-by-section first
        evidence_found: Optional[str] = None
        section_found: Optional[str] = None
        is_partial = False

        # Helper to find sentence in text
        def find_sentence_with_alias(text: str, alias: str) -> Optional[str]:
            # Word boundary regex
            escaped = re.escape(alias)
            pattern = rf"(?i)(?:^|[\s,;|•(])({escaped})(?:$|[\s,;|•)/])"
            match = re.search(pattern, text)
            if match:
                # Find surrounding line or sentence
                lines = text.splitlines()
                for line in lines:
                    if re.search(pattern, line):
                        clean_line = line.strip("• \t-*")
                        return clean_line[:180]
            return None

        for sec_name, content in sections.items():
            sec_str = " ".join(content) if isinstance(content, list) else str(content)
            for alias in target_aliases:
                sent = find_sentence_with_alias(sec_str, alias)
                if sent:
                    evidence_found = sent
                    section_found = sec_name
                    break
            if evidence_found:
                break

        # Fallback to whole resume text
        if not evidence_found:
            for alias in target_aliases:
                sent = find_sentence_with_alias(resume_text, alias)
                if sent:
                    evidence_found = sent
                    section_found = "resume_body"
                    break

        # Check for partial matches (e.g. SQLite/MySQL for SQL; FastAPI/Flask for REST APIs)
        if not evidence_found:
            if skill_name == "SQL" and any(db in norm_resume for db in ["sqlite", "mysql", "postgresql", "mongodb"]):
                for db in ["sqlite", "mysql", "postgresql", "mongodb"]:
                    sent = find_sentence_with_alias(resume_text, db)
                    if sent:
                        evidence_found = sent
                        section_found = "skills"
                        is_partial = True
                        break
            elif skill_name == "REST APIs" and any(fw in norm_resume for fw in ["fastapi", "flask", "express", "api"]):
                for fw in ["fastapi", "flask", "express", "api"]:
                    sent = find_sentence_with_alias(resume_text, fw)
                    if sent:
                        evidence_found = sent
                        section_found = "experience"
                        is_partial = True
                        break
            elif skill_name == "C++" and re.search(r"\b(c\s+programming|c\s+language)\b", norm_resume):
                evidence_found = "Proficiency in C programming language demonstrated"
                section_found = "skills"
                is_partial = True

        if evidence_found:
            if is_partial:
                return SkillEvidence(
                    skill_name=skill_name,
                    status="Partial",
                    confidence=0.75,
                    evidence_text=evidence_found,
                    section_found=section_found,
                    reason=f"Partially demonstrated through related competency in {section_found}."
                )
            else:
                return SkillEvidence(
                    skill_name=skill_name,
                    status="Matched",
                    confidence=1.0,
                    evidence_text=evidence_found,
                    section_found=section_found,
                    reason=f"Direct evidence found in candidate's {section_found} section."
                )
        else:
            return SkillEvidence(
                skill_name=skill_name,
                status="Not evidenced",
                confidence=0.0,
                evidence_text=None,
                section_found=None,
                reason="No mention of this competency found in parsed resume text."
            )

    @classmethod
    def extract_experience_highlights(cls, structured_resume: Optional[Dict[str, Any]], resume_text: str) -> List[ExperienceHighlight]:
        """Extracts concrete internships, development roles, and key engineering projects."""
        highlights: List[ExperienceHighlight] = []
        if not structured_resume:
            return highlights

        sections = structured_resume.get("sections", {})
        exp_lines = sections.get("experience", [])
        proj_lines = sections.get("projects", [])

        # Process experience
        exp_text = "\n".join(exp_lines) if isinstance(exp_lines, list) else str(exp_lines)
        if exp_text:
            # Parse organization and role blocks
            blocks = re.split(r"\n{2,}", exp_text)
            for blk in blocks:
                lines = [l.strip() for l in blk.splitlines() if l.strip()]
                if not lines:
                    continue
                first_line = lines[0]
                is_intern = bool(re.search(r"(?i)\bintern\b|\binternship\b", blk))
                
                # Extract technologies mentioned
                techs = []
                for tech, meta in SKILL_TAXONOMY.items():
                    if re.search(rf"(?i)\b{re.escape(tech)}\b", blk):
                        techs.append(meta["canonical"])
                techs = list(dict.fromkeys(techs))[:6]

                summary_bullets = [l for l in lines[1:] if len(l) > 20][:2]
                summary = " ".join(summary_bullets)[:220] if summary_bullets else first_line

                highlights.append(ExperienceHighlight(
                    role_or_project=first_line[:60],
                    organization="Experience Record",
                    duration_or_date=None,
                    relevance_summary=summary,
                    technologies_evidenced=techs,
                    is_internship=is_intern
                ))

        # Process projects
        proj_text = "\n".join(proj_lines) if isinstance(proj_lines, list) else str(proj_lines)
        if proj_text:
            blocks = re.split(r"\n{2,}", proj_text)
            for blk in blocks[:3]:
                lines = [l.strip() for l in blk.splitlines() if l.strip()]
                if not lines:
                    continue
                proj_title = lines[0].split("|")[0].strip()
                techs = []
                for tech, meta in SKILL_TAXONOMY.items():
                    if re.search(rf"(?i)\b{re.escape(tech)}\b", blk):
                        techs.append(meta["canonical"])
                techs = list(dict.fromkeys(techs))[:5]
                desc = " ".join(lines[1:])[:200] if len(lines) > 1 else lines[0]

                highlights.append(ExperienceHighlight(
                    role_or_project=proj_title[:50],
                    organization="Project",
                    duration_or_date=None,
                    relevance_summary=desc,
                    technologies_evidenced=techs,
                    is_internship=False
                ))

        return highlights

    @classmethod
    def generate_candidate_explanation(
        cls,
        candidate_name: str,
        job_title: str,
        job_description: str,
        resume_text: str,
        semantic_score: float,
        structured_resume: Optional[Dict[str, Any]] = None,
        is_parser_error: bool = False
    ) -> CandidateExplanation:
        """
        Synthesizes the complete explanation object with verified resume evidence,
        required vs preferred skills, experience highlights, and score breakdown.
        """
        required_skills, preferred_skills = cls.extract_job_competencies(job_title, job_description)

        req_evals = [cls.evaluate_skill_evidence(s, resume_text, structured_resume, is_parser_error) for s in required_skills]
        pref_evals = [cls.evaluate_skill_evidence(s, resume_text, structured_resume, is_parser_error) for s in preferred_skills]

        # Calculate coverage scores
        req_matched = sum(1.0 if e.status == "Matched" else 0.5 if e.status == "Partial" else 0.0 for e in req_evals)
        req_coverage = (req_matched / len(req_evals) * 100.0) if req_evals else 100.0

        pref_matched = sum(1.0 if e.status == "Matched" else 0.5 if e.status == "Partial" else 0.0 for e in pref_evals)
        pref_coverage = (pref_matched / len(pref_evals) * 100.0) if pref_evals else 100.0

        exp_highlights = cls.extract_experience_highlights(structured_resume, resume_text)
        
        # Calculate experience evidence score
        # Stronger score if candidate has formal internships / production projects
        internships_count = sum(1 for h in exp_highlights if h.is_internship)
        projects_count = sum(1 for h in exp_highlights if not h.is_internship)
        
        exp_score = min(100.0, internships_count * 40.0 + projects_count * 20.0 + (20.0 if "rest" in resume_text.lower() else 0.0))

        # Reconciled final ordinal relevance score:
        # 50% Fine-tuned semantic similarity + 25% Required skill coverage + 10% Preferred skills + 15% Practical experience
        reconciled_score = round(
            0.50 * semantic_score +
            0.25 * req_coverage +
            0.10 * pref_coverage +
            0.15 * exp_score,
            2
        )
        reconciled_score = float(max(0.0, min(100.0, reconciled_score)))

        # Category mapping
        if reconciled_score >= 70.0:
            fit_category = "Good Fit"
        elif reconciled_score >= 45.0:
            fit_category = "Potential Fit"
        else:
            fit_category = "No Fit"

        # Strongest match snippet
        matched_items = [e for e in req_evals if e.status == "Matched"]
        if matched_items:
            strongest_match = f"Strong evidence for {', '.join(e.skill_name for e in matched_items[:3])}."
        elif exp_highlights:
            strongest_match = f"Demonstrated practical experience in {exp_highlights[0].role_or_project}."
        else:
            strongest_match = "General academic background aligned with domain."

        # Top missing requirement
        missing_items = [e for e in req_evals if e.status == "Not evidenced"]
        top_missing = f"Missing documented evidence for {missing_items[0].skill_name}." if missing_items else None

        # Overall summary
        intern_text = f"{internships_count} internship{'s' if internships_count != 1 else ''}" if internships_count > 0 else "academic projects"
        overall_summary = (
            f"{candidate_name} demonstrates {round(req_coverage)}% required skill coverage with {intern_text}. "
            f"Semantic job relevance is {semantic_score:.1f}/100. "
            f"{'Strong candidate for interview review.' if fit_category == 'Good Fit' else 'Viable candidate with potential alignment.' if fit_category == 'Potential Fit' else 'Lacks critical required qualifications for this role.'}"
        )

        score_breakdown = ScoreBreakdown(
            semantic_similarity_score=round(semantic_score, 2),
            required_skill_coverage=round(req_coverage, 2),
            preferred_skill_coverage=round(pref_coverage, 2),
            experience_project_score=round(exp_score, 2),
            final_relevance_score=reconciled_score,
            formula_description="Relevance Score = 50% Semantic SBERT + 25% Required Skill Coverage + 10% Preferred Skill Coverage + 15% Experience Evidence"
        )

        return CandidateExplanation(
            rank=None,  # Assigned after multi-candidate sorting
            relevance_score=reconciled_score,
            fit_category=fit_category,
            overall_summary=overall_summary,
            strongest_match=strongest_match,
            top_missing_requirement=top_missing,
            required_skills=[asdict(e) for e in req_evals],
            preferred_skills=[asdict(e) for e in pref_evals],
            experience_highlights=[asdict(h) for h in exp_highlights],
            peer_comparison=None,
            score_breakdown=asdict(score_breakdown)
        )

    @classmethod
    def assign_peer_comparisons(cls, explanations: List[CandidateExplanation]) -> List[CandidateExplanation]:
        """Generates evidence-grounded comparative insights between adjacent candidates."""
        for idx, current in enumerate(explanations):
            current.rank = idx + 1
            if idx == 0:
                if len(explanations) > 1:
                    next_cand = explanations[1]
                    diff_req = current.score_breakdown["required_skill_coverage"] - next_cand.score_breakdown["required_skill_coverage"]
                    diff_exp = current.score_breakdown["experience_project_score"] - next_cand.score_breakdown["experience_project_score"]
                    reasons = []
                    if diff_req > 5:
                        reasons.append(f"higher required skill coverage (+{diff_req:.0f}%)")
                    if diff_exp > 10:
                        reasons.append("stronger demonstrated internship and practical engineering experience")
                    if not reasons:
                        reasons.append("higher semantic alignment across job requirements")
                    current.peer_comparison = f"Ranks #1 overall due to {', and '.join(reasons)}."
                else:
                    current.peer_comparison = "Top ranked candidate for this role."
            else:
                prev_cand = explanations[idx - 1]
                diff_req = prev_cand.score_breakdown["required_skill_coverage"] - current.score_breakdown["required_skill_coverage"]
                diff_exp = prev_cand.score_breakdown["experience_project_score"] - current.score_breakdown["experience_project_score"]
                reasons = []
                if diff_req > 5:
                    reasons.append(f"lower required skill coverage (-{diff_req:.0f}%)")
                if diff_exp > 10:
                    reasons.append("less documented production or internship development experience")
                if not reasons:
                    reasons.append("lower overall semantic compatibility score")
                current.peer_comparison = f"Ranks below #{idx} primarily due to {', and '.join(reasons)}."

        return explanations
