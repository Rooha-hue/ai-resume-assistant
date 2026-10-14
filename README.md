# 📄 ATS Resume Checker

Upload a resume (PDF, DOCX or TXT) and get an **estimated ATS score**, a score breakdown,
keyword gaps and specific suggestions for improvement. Built with **Streamlit** and a
**Google Gemini Flash** model.

> The score is an AI-generated estimate. Real ATS software differs between employers, so use it as guidance, not a guarantee.

## Features
- Reads PDF, DOCX and TXT resumes (up to 5 MB)
- Overall ATS score (0-100) built from 5 weighted categories
- Optional job description for targeted keyword matching
- Prioritised improvement list (High / Medium / Low) with rewrite suggestions
- Matched vs. missing keywords
- Downloadable Markdown report

## Project structure
```
app.py             # Streamlit app
requirements.txt   # Python dependencies
README.md
```

## Run locally
1. Get a free Gemini API key from https://aistudio.google.com/apikey
2. Install and run:
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
export GEMINI_API_KEY="your-key" # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
streamlit run app.py
```
You can also skip the environment variable and paste the key into the sidebar.

Optional: choose another model with `GEMINI_MODEL` (default `gemini-2.5-flash`) or the
sidebar "Gemini model" box. Use any Flash model id that your API key can access.

## Push to GitHub
```bash
cd your-project-folder
git init
printf "venv/\n__pycache__/\n.env\n.streamlit/secrets.toml\n" > .gitignore
git add .
git commit -m "Initial commit: ATS resume checker"
git branch -M main
# create an empty repo on github.com first (no README), then:
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```
Never commit your API key.

## Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** and pick your repository, branch `main`, and main file `app.py`.
3. Open **Advanced settings → Secrets** and add:
```toml
GEMINI_API_KEY = "your-key"
```
4. Click **Deploy**. Every `git push` to `main` redeploys automatically.

## Notes
- Scanned (image-only) PDFs have no text to read; export a text-based PDF or use DOCX.
- Resume text is sent to Google's Gemini API for analysis. Do not upload resumes you are not allowed to share.
