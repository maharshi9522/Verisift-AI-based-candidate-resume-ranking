import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

# Section Aliases Registry
# Maps canonical section keys to patterns/aliases
SECTION_REGISTRY: Dict[str, List[str]] = {
    "summary": [
        "professional summary", "summary", "career summary", "profile", "professional profile",
        "personal profile", "about me", "about", "career objective", "objective",
        "executive summary", "candidate overview", "introduction", "professional statement",
        "brief summary", "overview", "synopsis"
    ],
    "experience": [
        "work experience", "professional experience", "employment history", "work history",
        "career history", "industry experience", "relevant experience", "internship experience",
        "internships", "professional background", "experience", "relevant work",
        "employment", "positions held", "work record", "past roles", "work & experience",
        "consulting experience", "freelance experience", "practical experience"
    ],
    "education": [
        "education", "educational background", "academic background", "academic qualifications",
        "qualifications", "educational qualifications", "degrees", "academic history",
        "university education", "formal education", "schooling", "academics", "college"
    ],
    "skills": [
        "technical skills", "skills", "core skills", "key skills", "technical competencies",
        "core competencies", "competencies", "technology stack", "tech stack",
        "tools and technologies", "tools & technologies", "technologies", "programming skills",
        "software skills", "it skills", "areas of expertise", "technical expertise",
        "platforms and tools", "tools", "frameworks and tools", "programming languages",
        "software proficiencies", "technical proficiencies", "core abilities", "specializations"
    ],
    "projects": [
        "projects", "academic projects", "personal projects", "technical projects",
        "software projects", "key projects", "selected projects", "relevant projects",
        "project experience", "research projects", "major projects", "capstone project",
        "final year project", "portfolio projects", "open source projects", "notable projects",
        "selected engineering projects", "hands-on projects", "hackathon projects"
    ],
    "certifications": [
        "certifications", "certificates", "professional certifications", "technical certifications",
        "courses", "online courses", "training and certifications", "training & certifications",
        "training", "professional development", "workshops and training", "licenses and certifications",
        "licenses & certifications", "continuing education", "credentials", "accreditations"
    ],
    "achievements": [
        "achievements", "key achievements", "accomplishments", "awards", "honors", "honours",
        "recognition", "academic achievements", "professional achievements", "awards and recognition",
        "distinctions", "scholarships", "competitive achievements", "accolades", "milestones"
    ],
    "research_publications": [
        "research", "research experience", "research interests", "publications", "selected publications",
        "research publications", "papers", "conference papers", "journal publications",
        "scientific publications", "patent", "patents", "thesis", "dissertation", "research work"
    ],
    "leadership_volunteering": [
        "leadership", "leadership experience", "positions of responsibility", "responsibilities",
        "extracurricular activities", "extracurriculars", "activities", "volunteering",
        "volunteer experience", "community service", "student activities", "clubs and societies",
        "campus involvement", "student leadership", "social impact", "volunteer work"
    ],
    "languages": [
        "languages", "language proficiency", "languages known", "spoken languages"
    ],
    "coursework": [
        "coursework", "relevant coursework", "core subjects", "relevant subjects",
        "academic coursework", "major courses", "electives"
    ],
    "interests": [
        "interests", "hobbies", "personal interests", "extracurricular interests", "passions"
    ]
}

# Compile alias lookup dictionary for fast matching
NORM_TO_CANONICAL: Dict[str, str] = {}
for canonical, aliases in SECTION_REGISTRY.items():
    for alias in aliases:
        norm = re.sub(r"[^a-z0-9]", "", alias.lower())
        NORM_TO_CANONICAL[norm] = canonical

@dataclass
class CandidateIdentity:
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    portfolio_url: Optional[str] = None

@dataclass
class ResumeChunk:
    chunk_id: str
    section: str
    text: str
    token_estimate: int
    char_start: int
    char_end: int

@dataclass
class StructuredResume:
    identity: CandidateIdentity
    sections: Dict[str, List[str]]
    chunks: List[ResumeChunk]
    raw_text: str
    cleaned_text: str
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity": asdict(self.identity),
            "sections": self.sections,
            "chunks": [asdict(c) for c in self.chunks],
            "raw_text": self.raw_text,
            "cleaned_text": self.cleaned_text,
            "extraction_metadata": self.extraction_metadata
        }

class ResumeParser:
    """
    Robust, section-aware resume parser and structured feature extractor.
    Extracts candidate contact details, recognizes section boundaries,
    cleans noisy formatting artifacts, and divides documents into semantic chunks.
    """

    @classmethod
    def extract_identity(cls, raw_text: str, filename: Optional[str] = None) -> Tuple[CandidateIdentity, List[str]]:
        """
        Extracts contact details and full name conservatively from header lines.
        Returns: (CandidateIdentity, remaining_lines_without_contact_noise)
        """
        identity = CandidateIdentity()
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

        # 1. Regex for Email
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"
        email_match = re.search(email_pattern, raw_text)
        if email_match:
            identity.email = email_match.group(0).strip().lower()

        # 2. Regex for Phone (US, International, standard formats)
        phone_pattern = r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
        phone_match = re.search(phone_pattern, raw_text)
        if phone_match:
            phone_candidate = phone_match.group(0).strip()
            # Verify minimum digits to avoid date false positives
            if sum(c.isdigit() for c in phone_candidate) >= 10:
                identity.phone = phone_candidate

        # 3. Regex for LinkedIn
        linkedin_pattern = r"(?:https?://)?(?:www\.)?linkedin\.com/in/([A-Za-z0-9_-]+)"
        li_match = re.search(linkedin_pattern, raw_text, re.IGNORECASE)
        if li_match:
            identity.linkedin_url = f"https://linkedin.com/in/{li_match.group(1)}"

        # 4. Regex for GitHub
        github_pattern = r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9_-]+)"
        gh_match = re.search(github_pattern, raw_text, re.IGNORECASE)
        if gh_match:
            username = gh_match.group(1).lower()
            if username not in ("pulls", "issues", "features", "explore"):
                identity.github_url = f"https://github.com/{username}"

        # 5. Regex for Portfolio / Personal Website
        portfolio_pattern = r"(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+\.(?:dev|me|io|portfolio|app|ai|org|com))(?:/[^\s]*)?"
        for m in re.finditer(portfolio_pattern, raw_text, re.IGNORECASE):
            domain = m.group(1).lower()
            if not any(skip in domain for skip in ("linkedin", "github", "example", "stanford", "berkeley", "google", "gmail")):
                identity.portfolio_url = m.group(0)
                break

        # 6. Regex for Location (City, State / Country)
        for line in lines[:6]:
            # Check segments separated by |, •, or commas
            segments = re.split(r"\||•", line)
            for seg in segments:
                seg_clean = seg.strip()
                # Match standard "City, ST" (e.g. "San Francisco, CA", "Seattle, WA") or "City, Country"
                m = re.match(r"^([A-Z][a-zA-Z\s.-]+,\s*(?:[A-Z]{2}|[A-Z][a-zA-Z]+))$", seg_clean)
                if m:
                    cand_loc = m.group(1).strip()
                    if not any(term in cand_loc.lower() for term in ("university", "experience", "skills", "summary", "engineer", "developer")):
                        identity.location = cand_loc
                        break
            if identity.location:
                break

        # 7. Conservative Name Extraction
        # Look at the first 3 non-empty lines before any section heading
        name_candidate = None
        for line in lines[:4]:
            # Clean separators and titles after '-', '|', '•'
            clean_l = re.split(r"[-|•,]", line)[0].strip()
            words = clean_l.split()
            if 2 <= len(words) <= 4:
                    banned = ("resume", "curriculum", "vitae", "summary", "engineer", "developer", "experience",
                              "skills", "technical", "education", "comptia", "certified", "analyst", "specialist",
                              "training", "project", "associate", "manager")
                    if not any(k in clean_l.lower() for k in banned) and "+" not in clean_l:
                        if not cls._match_glued_heading(clean_l):
                            if all(w.isalpha() and w[0].isupper() for w in words if w):
                                name_candidate = clean_l
                                break

        if name_candidate:
            identity.full_name = name_candidate
        elif filename:
            # Fallback cleanly to filename if informative e.g. "Alex_Rivera_ML_Engineer.pdf" -> "Alex Rivera"
            stem = re.sub(r"[._](pdf|docx|doc)$", "", filename, flags=re.IGNORECASE)
            name_parts = re.split(r"[._-]", stem)
            valid_parts = [p for p in name_parts if p.isalpha() and p.lower() not in ("ml", "resume", "cv", "engineer", "developer", "fullstack")]
            if 2 <= len(valid_parts) <= 3:
                identity.full_name = " ".join(p.capitalize() for p in valid_parts)

        return identity, lines

    @classmethod
    def identify_heading(cls, line: str) -> Optional[str]:
        """
        Determines whether a line is a section heading.
        Returns the canonical section category key if recognized, or 'other' if an unknown heading, or None.
        """
        line_clean = line.strip()
        if not line_clean:
            return None

        # Exclude ordinary long sentences (> 60 characters or > 8 words)
        words = line_clean.split()
        if len(line_clean) > 60 or len(words) > 8:
            return None

        # Strip markdown markers (#, **, __) and trailing colons/dashes
        stripped = re.sub(r"^[#*\-_~]+\s*", "", line_clean)
        stripped = re.sub(r"[\s:*_~-]+$", "", stripped).strip()
        if not stripped:
            return None

        # Check for combined headings (e.g. "Skills & Tools", "Projects / Research")
        # Split on '&', 'and', '/', '|'
        sub_headings = re.split(r"\s*(?:&|and|/|\|)\s*", stripped, flags=re.IGNORECASE)
        for sub in sub_headings:
            norm_sub = re.sub(r"[^a-z0-9]", "", sub.lower())
            if norm_sub in NORM_TO_CANONICAL:
                return NORM_TO_CANONICAL[norm_sub]

        # Normalized full heading check
        norm = re.sub(r"[^a-z0-9]", "", stripped.lower())
        if norm in NORM_TO_CANONICAL:
            return NORM_TO_CANONICAL[norm]

        # Heading heuristic cues: ALL CAPS (>= 4 letters) or short title before colon
        if stripped.isupper() and len(words) <= 4 and len(stripped) >= 4:
            # Check if likely a section heading
            if not any(char.isdigit() for char in stripped):
                return "other"

        if line_clean.endswith(":") and len(words) <= 4:
            if not any(char.isdigit() for char in stripped):
                return "other"

        return None

    @classmethod
    def _match_glued_heading(cls, line: str) -> Optional[Tuple[str, str]]:
        """
        Detects headings that are directly concatenated with body text
        (common in OCR, PDF streams, and web datasets like 'SummaryI started...').
        """
        # Collect aliases sorted by length descending
        if not hasattr(cls, "_glued_regex"):
            all_aliases = []
            for canonical, aliases in SECTION_REGISTRY.items():
                for alias in aliases:
                    all_aliases.append((alias, canonical))
            all_aliases.sort(key=lambda x: len(x[0]), reverse=True)
            pattern_str = r"^(" + "|".join(re.escape(a[0]) for a in all_aliases) + r")(?=[A-Z0-9\s:•\-])"
            cls._glued_regex = re.compile(pattern_str, re.IGNORECASE)
            cls._alias_to_canonical = {re.sub(r"[^a-z0-9]", "", a[0].lower()): a[1] for a in all_aliases}

        m = cls._glued_regex.match(line)
        if m:
            matched_phrase = m.group(1)
            norm_phrase = re.sub(r"[^a-z0-9]", "", matched_phrase.lower())
            canonical = cls._alias_to_canonical.get(norm_phrase, "other")
            rest = line[len(matched_phrase):].strip()
            return canonical, rest
        return None

    @classmethod
    def clean_text_block(cls, text: str) -> str:
        """Removes pagination noise, repeated headers/footers, and broken hyphenation."""
        # Remove page numbers like 'Page 1 of 2', 'Page 2', '- 1 -'
        text = re.sub(r"(?i)\bpage\s+\d+(\s+of\s+\d+)?\b", "", text)
        text = re.sub(r"^\s*[-—]\s*\d+\s*[-—]\s*$", "", text, flags=re.MULTILINE)

        # Fix broken hyphenated line wraps: e.g. "multi-\nprocessing" -> "multiprocessing"
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

        # Normalize multiple spaces and non-standard bullet characters
        text = re.sub(r"[•●▪◆➢]\s*", "- ", text)
        text = re.sub(r"[ \t]+", " ", text)

        # Collapse excess empty lines
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    @classmethod
    def parse_resume(cls, raw_text: str, filename: Optional[str] = None) -> StructuredResume:
        """
        Parses raw resume text into a StructuredResume containing extracted identity,
        canonical sections, cleaned text, and semantic chunks.
        """
        if not raw_text or not raw_text.strip():
            empty_id = CandidateIdentity()
            return StructuredResume(
                identity=empty_id,
                sections={},
                chunks=[],
                raw_text="",
                cleaned_text="",
                extraction_metadata={"total_sections": 0, "total_chunks": 0}
            )

        # 1. Extract contact details
        identity, lines = cls.extract_identity(raw_text, filename)

        # 2. Section Segmentation
        sections: Dict[str, List[str]] = {cat: [] for cat in SECTION_REGISTRY.keys()}
        sections["other"] = []

        current_section: Optional[str] = None
        header_lines: List[str] = []

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            detected_heading = cls.identify_heading(line_str)
            if detected_heading:
                current_section = detected_heading
                continue

            glued = cls._match_glued_heading(line_str)
            if glued:
                current_section, rest_text = glued
                if rest_text:
                    sections[current_section].append(rest_text)
                continue

            if current_section is None:
                header_lines.append(line_str)
            else:
                sections[current_section].append(line_str)

        # If summary was not explicitly tagged, look for headline/objective in non-contact header lines
        if not sections["summary"] and header_lines:
            non_contact = []
            for hl in header_lines:
                if (identity.full_name and identity.full_name.lower() in hl.lower()) or \
                   (identity.email and identity.email.lower() in hl.lower()) or \
                   (identity.phone and identity.phone in hl) or \
                   (identity.location and identity.location.lower() in hl.lower()) or \
                   ("github.com" in hl.lower() or "linkedin.com" in hl.lower()):
                    continue
                if len(hl.split()) >= 4:
                    non_contact.append(hl)
            if non_contact:
                sections["summary"].extend(non_contact)

        # Clean empty sections
        cleaned_sections: Dict[str, List[str]] = {}
        for sec_name, sec_lines in sections.items():
            if sec_lines:
                sec_text = cls.clean_text_block("\n".join(sec_lines))
                if sec_text:
                    cleaned_sections[sec_name] = [sec_text]

        # 3. Create Semantic Chunks (Part C requirement)
        chunks: List[ResumeChunk] = []
        chunk_idx = 0
        overall_cleaned_parts = []

        # Process sections in priority order
        section_priority = [
            "skills", "experience", "projects", "summary", "education",
            "certifications", "achievements", "research_publications",
            "leadership_volunteering", "coursework", "languages", "other"
        ]

        char_offset = 0
        for sec_name in section_priority:
            if sec_name not in cleaned_sections:
                continue

            sec_content = "\n\n".join(cleaned_sections[sec_name])
            overall_cleaned_parts.append(f"[{sec_name.upper()}]\n{sec_content}")

            # Chunk section content if long
            # MiniLM supports 256 tokens (~180-200 words).
            # Split by paragraphs or bullet blocks, target max 180 words per chunk.
            paragraphs = [p.strip() for p in sec_content.split("\n\n") if p.strip()]
            current_chunk_words: List[str] = []

            for p in paragraphs:
                words = p.split()
                if len(current_chunk_words) + len(words) <= 180:
                    current_chunk_words.extend(words)
                else:
                    if current_chunk_words:
                        chunk_text = " ".join(current_chunk_words)
                        chunks.append(ResumeChunk(
                            chunk_id=f"chk_{chunk_idx:03d}_{sec_name}",
                            section=sec_name,
                            text=f"[{sec_name.upper()}] {chunk_text}",
                            token_estimate=int(len(chunk_text.split()) * 1.3),
                            char_start=char_offset,
                            char_end=char_offset + len(chunk_text)
                        ))
                        char_offset += len(chunk_text) + 1
                        chunk_idx += 1
                    current_chunk_words = words

            if current_chunk_words:
                chunk_text = " ".join(current_chunk_words)
                chunks.append(ResumeChunk(
                    chunk_id=f"chk_{chunk_idx:03d}_{sec_name}",
                    section=sec_name,
                    text=f"[{sec_name.upper()}] {chunk_text}",
                    token_estimate=int(len(chunk_text.split()) * 1.3),
                    char_start=char_offset,
                    char_end=char_offset + len(chunk_text)
                ))
                char_offset += len(chunk_text) + 1
                chunk_idx += 1

        cleaned_text = "\n\n".join(overall_cleaned_parts)

        metadata = {
            "total_sections_found": len(cleaned_sections),
            "sections_present": list(cleaned_sections.keys()),
            "total_chunks": len(chunks),
            "has_email": bool(identity.email),
            "has_phone": bool(identity.phone),
            "has_name": bool(identity.full_name),
            "has_linkedin": bool(identity.linkedin_url),
            "has_github": bool(identity.github_url)
        }

        return StructuredResume(
            identity=identity,
            sections=cleaned_sections,
            chunks=chunks,
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            extraction_metadata=metadata
        )
