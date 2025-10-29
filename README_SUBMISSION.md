# Simplified Revenue Leakage Dashboard

A streamlined interface for detecting and fixing revenue leakage issues.

## Demo Video

Watch or download the full demo:

- [Watch the demo (MOV)](media/demo.mov)
- [Watch the demo (GIF)](media/demo.gif)

HTML embed (may not render in all viewers):

<video src="media/demo.mov" controls width="800"></video>

## Features

### 🎯 Simple User Flow
- **Split-screen view** comparing billing plans and invoices
- **Three main tabs**: Billing Plans, Invoices, and Leakage Detection
- **One-click analysis** to detect revenue leakage issues
- **Filterable results** with severity and type filters

### 🔍 Leakage Detection
- **Automatic detection** of missing invoices, overbilling, and currency conversion issues
- **Severity-based filtering** (High, Medium, Low)
- **Type-based filtering** for different issue types
- **Real-time metrics** showing issue counts by severity

### 💡 Smart Fixes
- **View Details** button for each issue showing evidence and calculations
- **Apply Fix** button that automatically creates the appropriate corrective action
- **Evidence display** showing expected vs actual amounts and calculations
- **Sandbox integration** - all fixes are saved to the sandbox environment

### 💬 Chat Interface
- **Simple chat** in the sidebar for asking questions about revenue leakage
- **Context-aware responses** based on the current analysis

## How to Use

1. **Start the application**:
   ```bash
   streamlit run app.py
   ```

2. **Select a mission** from the sidebar (e.g., "Detect revenue leakage")

3. **Choose a customer filter** (or "All" to see all customers)

4. **Click "Analyze Revenue"** to run the leakage detection

5. **View results** in the "Leakage Detection" tab:
   - Use filters to narrow down results
   - Click "View Details" to see evidence and proposed fixes
   - Click "Apply Fix" to automatically create corrective actions

6. **Chat with the agent** using the sidebar chat interface

## Key Improvements

- ✅ **Simplified interface** - removed complex multi-step workflows
- ✅ **Split-screen comparison** - easy to compare plans vs invoices
- ✅ **One-click analysis** - single button to detect all issues
- ✅ **Filterable results** - easy to focus on specific issues
- ✅ **Evidence-based popups** - clear calculations and reasoning
- ✅ **One-click fixes** - automatic creation of corrective actions
- ✅ **Sandbox integration** - all changes saved with audit trail

## File Structure

- `app.py` - Main Streamlit application
- `agent.py` - AI agent for analysis and reasoning
- `tools.py` - Tools for data manipulation and fix creation
- `data/` - JSON files containing billing plans, invoices, etc.
- `sandbox/` - Applied actions and audit log

## Requirements

- Python 3.8+
- Streamlit
- LangChain/LangGraph
- OpenAI API key (set in .env file)

## Setup

1. Create and activate a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   - Create a `.env` file in the project root with your OpenAI key:
     ```bash
     echo "OPENAI_API_KEY=your_key_here" > .env
     ```
   - Optional: set any other keys required by your environment.

4. Run the app:
   ```bash
   streamlit run app.py
   ```

5. Data and sandbox:
   - Input data lives in `data/` (JSON files)
   - Fixes and audit logs write to `sandbox/`

Troubleshooting:
- If Streamlit can’t find the environment variables, ensure the `.env` file exists at the project root.
- If imports fail, confirm your virtual environment is activated and `pip install -r requirements.txt` completed successfully.



## Next Steps

The dashboard now provides a much simpler user experience while maintaining all the powerful analysis capabilities. Users can quickly identify issues, understand the evidence, and apply fixes with minimal clicks.
