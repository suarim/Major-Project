
## 1. Environment Configuration

### A. Create Environment Files

1. **Main Configuration (.env)**
   Create `.env` in your project root with:
   ```env
   PORT=5000
   ```

2. **Gemini API Key (server/scripts/.env)**
   Create `server/scripts/.env` with:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```

## 2. Node.js Setup

Install dependencies:
```bash
npm install
```

## 3. Python Virtual Environments(Both venv creation in root directory only)

### A. Signs Environment (for MediaPipe)
```bash
# Create environment
python -m venv signs

# Activate (Windows)
.\signs\Scripts\activate
# Or (macOS/Linux)
source signs/bin/activate

# Install dependencies
pip install -r signs.txt
```

### B. Gemini Environment (for API Processing)
```bash
# Create environment
python -m venv gemini-env

# Activate (Windows)
.\gemini-env\Scripts\activate
# Or (macOS/Linux)
source gemini-env/bin/activate

# Install dependencies
pip install -r gemini.txt
```

## 4. Running the Application

1. First activate the signs environment:
   ```bash
   .\signs\Scripts\activate  # Windows
   source signs/bin/activate  # macOS/Linux
   ```

2. Then start the development server:
   ```bash
   npm run dev
   ```

## 🔧 Troubleshooting

- **Activation Issues**: On Windows, check for `Activate.ps1` (uppercase A) if `activate.ps1` doesn't work
- **Missing Dependencies**: Ensure you're in the correct virtual environment before installing requirements
- **API Errors**: Verify your `GEMINI_API_KEY` is properly set in `server/scripts/.env`
