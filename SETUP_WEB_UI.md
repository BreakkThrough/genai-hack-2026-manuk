# Web UI Setup Guide

This guide walks you through setting up and running the React + FastAPI interactive pipeline UI.

## Prerequisites

- Python 3.10+
- Node.js 18+ and npm
- Azure Document Intelligence resource
- Azure OpenAI resource with GPT-4o deployment

## Step 1: Backend Setup

### Install Python Dependencies

```bash
# Activate your virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Install updated dependencies (includes FastAPI)
pip install -r requirements.txt
```

### Configure Environment Variables

Make sure your `.env` file contains the required Azure credentials:

```env
AZURE_DI_ENDPOINT=https://your-di-resource.cognitiveservices.azure.com/
AZURE_DI_KEY=your-di-key
AZURE_OPENAI_ENDPOINT=https://your-openai-resource.openai.azure.com/
AZURE_OPENAI_KEY=your-openai-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

### Start the Backend

```bash
# Option 1: Direct command
uvicorn app.api:app --reload --port 8000

# Option 2: Using startup script
# Windows:
start_backend.bat

# macOS/Linux:
./start_backend.sh
```

The backend will be available at `http://localhost:8000`

You can verify it's running by visiting `http://localhost:8000/docs` to see the interactive API documentation.

## Step 2: Frontend Setup

### Install Node Dependencies

```bash
cd frontend
npm install
```

### Start the Frontend

```bash
# Option 1: Direct command
npm run dev

# Option 2: Using startup script
# Windows:
start_frontend.bat

# macOS/Linux:
./start_frontend.sh
```

The frontend will be available at `http://localhost:5173`

## Step 3: Using the Web UI

1. **Upload Files**
   - Click "Select PDF Drawing" to choose your engineering drawing PDF
   - Click "Select STEP File" to choose your 3D model file
   - Select the appropriate units (inch or mm)
   - Click "Upload & Initialize"

2. **Run Pipeline Layers**
   - Click "Run Layer 1 (OCR)" to extract text with Azure DI
   - Click "Run Layer 2 (Vision)" to extract hole annotations with GPT-4o
   - Layer 3 (STEP parsing) runs automatically on upload
   - Click "Run Layer 4 (Correlation)" to match annotations with 3D features

3. **View Results**
   - The drawing viewer shows your PDF with clickable annotations
   - The right panel shows layer-specific output
   - After Layer 4, a mapping table appears where you can edit matches
   - The JSON export panel lets you download the final result

4. **Edit Mappings (Optional)**
   - In the mapping table, use the dropdown to reassign hole IDs
   - Click "Save Changes" to update the correlation
   - Export the updated JSON

## Troubleshooting

### Backend won't start
- Make sure you've activated the virtual environment
- Check that all dependencies are installed: `pip list | grep fastapi`
- Verify your `.env` file has all required variables

### Frontend won't start
- Make sure you're in the `frontend` directory
- Run `npm install` again if packages are missing
- Check Node version: `node --version` (should be 18+)

### CORS errors
- Make sure the backend is running on port 8000
- Check that the frontend is accessing `http://localhost:8000`
- The backend is configured to allow requests from `localhost:3000` and `localhost:5173`

### Images not loading
- Check the browser console for errors
- Verify the session ID is valid
- Make sure the PDF was uploaded successfully

## Development Notes

### Hot Reload

Both the backend and frontend support hot reload:
- Backend: Changes to Python files automatically restart the server
- Frontend: Changes to React files automatically refresh the browser

### API Documentation

Visit `http://localhost:8000/docs` to see the interactive Swagger UI documentation for all API endpoints.

### Component Structure

```
frontend/src/
├── components/
│   ├── Sidebar.tsx              # File upload and pipeline control
│   ├── PipelineStepBar.tsx      # Progress indicator
│   ├── DrawingViewer.tsx        # 2D drawing with annotations
│   ├── LayerOutputPanel.tsx     # Layer-specific output
│   ├── MappingTable.tsx         # Editable mappings
│   └── JSONExportPanel.tsx      # JSON preview and download
├── App.tsx                      # Main application
├── api.ts                       # Backend API client
└── types.ts                     # TypeScript definitions
```

## Production Build

To build the frontend for production:

```bash
cd frontend
npm run build
```

The optimized build will be in `frontend/dist/`.

To serve the production build:

```bash
npm run preview
```

For production deployment, you can serve the static files from `frontend/dist/` using any web server (nginx, Apache, etc.) and configure it to proxy API requests to the FastAPI backend.
