import pytest
from app.services.resume_parser import ResumeParser, CandidateIdentity, StructuredResume

def test_conventional_resume_parsing():
    text = """
    Alex Rivera
    alex.rivera@example.com | (415) 555-0199 | San Francisco, CA
    linkedin.com/in/alexrivera | github.com/arivera

    Professional Summary
    Senior Machine Learning Engineer with 6+ years of production deep learning experience.
    Led recommendation system redesign resulting in 28% increase in user engagement.

    Work Experience
    Senior ML Engineer | TechCorp Inc. | 2021 - Present
    - Architected two-tower semantic search models serving 10M queries daily.
    - Reduced latency by 45% using TensorRT and ONNX runtime optimization.

    ML Engineer | DataFlow Systems | 2018 - 2021
    - Built computer vision pipelines for document layout analysis.
    - Fine-tuned BERT and RoBERTa models for text classification achieving 94.2% accuracy.

    Education
    Master of Science in Computer Science | Stanford University | 2018
    Bachelor of Science in Electrical Engineering | UC Berkeley | 2016

    Technical Skills
    Languages: Python, C++, SQL, TypeScript
    Frameworks: PyTorch, TensorFlow, Hugging Face, FastAPI
    Tools: Docker, Kubernetes, AWS, Ray
    """
    parsed = ResumeParser.parse_resume(text)
    assert parsed.identity.full_name == "Alex Rivera"
    assert parsed.identity.email == "alex.rivera@example.com"
    assert parsed.identity.phone == "(415) 555-0199"
    assert "San Francisco" in parsed.identity.location
    assert "linkedin.com/in/alexrivera" in parsed.identity.linkedin_url
    assert "github.com/arivera" in parsed.identity.github_url

    assert "summary" in parsed.sections
    assert "experience" in parsed.sections
    assert "education" in parsed.sections
    assert "skills" in parsed.sections
    assert len(parsed.chunks) >= 4
    # Check that quantitative metrics are preserved
    assert any("28%" in c.text for c in parsed.chunks)
    assert any("45%" in c.text for c in parsed.chunks)

def test_resume_no_explicit_name_heading():
    text = """
    david.k@domain.org | +1 650 555 4321
    Seattle, WA

    Summary
    Backend developer specializing in distributed systems.

    Skills
    Go, Rust, PostgreSQL, gRPC
    """
    # Name is not explicitly given in text; should fallback to filename if supplied
    parsed_with_file = ResumeParser.parse_resume(text, filename="David_Kowalski_Resume.pdf")
    assert parsed_with_file.identity.full_name == "David Kowalski"
    assert parsed_with_file.identity.email == "david.k@domain.org"

def test_unconventional_and_combined_headings():
    text = """
    Elena Rostova
    elena@example.com

    Work & Experience / Internships
    - Software Intern at Meta (Summer 2022)
    - Research Intern at MIT CSAIL (2021)

    Academic Qualifications & Degrees
    - B.S. in Computer Science, MIT (GPA: 3.9/4.0)

    Tools | Technologies | Core Competencies
    - Python, PyTorch, C++, Linux, Git

    Selected Projects & Research Publications
    - Graph Neural Networks for Molecule Generation (NeurIPS 2023 workshop)
    """
    parsed = ResumeParser.parse_resume(text)
    assert parsed.identity.full_name == "Elena Rostova"
    assert "experience" in parsed.sections
    assert "education" in parsed.sections
    assert "skills" in parsed.sections
    # Projects & publications can be in projects or research_publications
    assert ("projects" in parsed.sections) or ("research_publications" in parsed.sections)

def test_missing_sections_and_skills_only():
    text = """
    Code Ninja
    ninja@coders.io

    Technical Competencies
    Python, Django, FastAPI, Celery, Redis, PostgreSQL, React

    Personal Projects
    - Built automated code reviewer using LLMs with 500+ GitHub stars.
    - Developed real-time collaborative code editor in WebSockets.
    """
    parsed = ResumeParser.parse_resume(text)
    assert "skills" in parsed.sections
    assert "projects" in parsed.sections
    assert "education" not in parsed.sections
    assert len(parsed.chunks) >= 2

def test_multipage_pagination_and_repeated_headers():
    text = """
    Samantha Wells | sam@wells.dev | Page 1 of 3
    Senior Site Reliability Engineer

    Summary
    Experienced SRE managing multi-region Kubernetes clusters.

    --- Page 2 of 3 ---
    Samantha Wells | sam@wells.dev
    Work Experience
    Lead SRE at CloudScale Inc. (2020 - Present)
    - Managed 50+ EKS clusters with 99.99% uptime.

    --- Page 3 of 3 ---
    Samantha Wells | sam@wells.dev
    Education
    BS in Information Systems, Purdue University
    """
    parsed = ResumeParser.parse_resume(text)
    # Ensure repeated headers like "Page 1 of 3" are stripped from chunks
    for chunk in parsed.chunks:
        assert "Page 1 of 3" not in chunk.text
        assert "Page 2 of 3" not in chunk.text
        assert "Page 3 of 3" not in chunk.text

def test_unfamiliar_section_preserved():
    text = """
    Dr. Robert Lang
    robert@biotech.org

    Patents and Intellectual Property
    - US Patent 11,234,567: Novel CRISPR delivery mechanism for in-vivo gene editing.

    Summary
    Computational biologist.
    """
    parsed = ResumeParser.parse_resume(text)
    # The unfamiliar section should be preserved in research_publications or other
    assert ("research_publications" in parsed.sections) or ("other" in parsed.sections)
    combined = " ".join([c.text for c in parsed.chunks])
    assert "CRISPR delivery mechanism" in combined

def test_missing_contact_info_conservative():
    text = """
    A candidate with no phone, no email, and no links.
    Only raw text describing their programming journey.

    Skills
    Python, Java, C
    """
    parsed = ResumeParser.parse_resume(text)
    assert parsed.identity.email is None
    assert parsed.identity.phone is None
    assert parsed.identity.linkedin_url is None
    assert parsed.identity.github_url is None
    assert "skills" in parsed.sections

def test_unicode_and_special_characters():
    text = """
    François Müller • München, Germany
    francois.muller@example.eu | +49 89 1234567

    Profil Professionnel
    Ingénieur en intelligence artificielle • Spécialité NLP & LLMs.
    Résultats: réduction de latence de 35% — précision améliorée de 4.5 points.

    Compétences Clés
    Python 3.11, PyTorch, HuggingFace Transformers, Docker™
    """
    parsed = ResumeParser.parse_resume(text)
    assert "François Müller" in parsed.identity.full_name
    assert "francois.muller@example.eu" == parsed.identity.email
    assert len(parsed.chunks) >= 1
    # Check that accents and special characters are preserved
    assert any("réduction de latence" in c.text for c in parsed.chunks)

def test_empty_and_whitespace_only_resume():
    parsed = ResumeParser.parse_resume("   \n\n   ")
    assert parsed.identity.full_name is None
    assert parsed.identity.email is None
    assert len(parsed.chunks) == 0
    assert parsed.to_dict()["raw_text"] == ""
