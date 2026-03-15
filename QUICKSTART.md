# Quick Start Guide

Get the React + FastAPI web UI running in 3 minutes.

## 1. Install Backend Dependencies

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Install Python packages
pip install -r requirements.txt
```

## 2. Configure Azure Credentials

Edit `.env` with your Azure credentials:

```env
AZURE_DI_ENDPOINT=https://your-di-resource.cognitiveservices.azure.com/
AZURE_DI_KEY=your-di-key
AZURE_OPENAI_ENDPOINT=https://your-openai-resource.openai.azure.com/
AZURE_OPENAI_KEY=your-openai-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

## 3. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

## 4. Start Both Servers

**Terminal 1 - Backend:**
```bash
uvicorn app.api:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

## 5. Open the Web UI

Open your browser to: `http://localhost:5173`

## Using the Pipeline

1. Click "Select PDF Drawing" and choose a drawing file
2. Click "Select STEP File" and choose a 3D model file
3. Select units (inch or mm)
4. Click "Upload & Initialize"
5. Click "Run Layer 1 (OCR)" and wait for completion
6. Click "Run Layer 2 (Vision)" and wait for completion
7. Layer 3 runs automatically on upload
8. Click "Run Layer 4 (Correlation)" to create mappings
9. View and edit mappings in the table
10. Download the final JSON result

## Test with NIST Dataset

If you have the NIST FTC dataset in the `dataset/` folder, you can test with:

- PDF: `dataset/nist_ftc_06_asme1_rd.pdf`
- STEP: `dataset/nist_ftc_06_asme1_rd.stp`
- Units: inch

Expected results: 10-15 hole annotations matched to 3D features.

## Troubleshooting

**Backend won't start:**
- Check virtual environment is activated
- Run `pip install -r requirements.txt` again

**Frontend won't start:**
- Make sure you're in the `frontend` directory
- Run `npm install` again
- Check Node version: `node --version` (need 18+)

**Can't upload files:**
- Make sure backend is running on port 8000
- Check browser console for CORS errors
- Verify `.env` has valid Azure credentials

**Vision layer takes a long time:**
- Normal! GPT-4o multi-pass extraction takes 30-90 seconds
- Check backend terminal for progress logs
- Each page is processed multiple times for maximum accuracy

## API Documentation

Visit `http://localhost:8000/docs` to see interactive API documentation.
