# Drawing-3D Hole Linker - React Frontend

Interactive web UI for the GenAI-powered drawing-to-3D hole feature linking pipeline.

## Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Architecture

The frontend is a single-page React application built with:

- **React 19** with TypeScript
- **Material-UI (MUI)** for components
- **Vite** for build tooling
- **Axios** for API communication

## Features

- Step-by-step pipeline execution with visual progress indicator
- 2D drawing viewer with page navigation
- Clickable annotation overlays
- Context-sensitive output panel for each pipeline layer
- Editable mapping table with dropdown selectors for reassigning hole matches
- JSON export with preview and download

## Backend Connection

The frontend expects the FastAPI backend to be running at `http://localhost:8000`.

Start the backend first:

```bash
cd ..
uvicorn app.api:app --reload --port 8000
```

Then start the frontend:

```bash
npm run dev
```

Open `http://localhost:5173` in your browser.
