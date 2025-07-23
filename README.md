# TacticPlanner

TacticPlanner is a web-based application designed to optimize marketing mix strategies for pharmaceutical companies. It leverages historical data and advanced AI models to provide actionable insights and recommendations.

## Features
- **User Authentication**: Sign up and log in to access personalized features.
- **AI-Powered Recommendations**: Uses advanced language models to generate marketing mix strategies.
- **Data-Driven Insights**: Analyzes historical data to provide context-aware suggestions.
- **Interactive Chat Interface**: Engage with the AI assistant for real-time recommendations.

## Project Structure
```
TacticPlanner/
│
├── app.py                   # Main application logic
├── templates/               # HTML templates
│   ├── signup.html
│   ├── chat.html
├── static/                  # Static files (CSS, JS, images)
│   ├── style.css
│   ├── chat.css
├── data/                    # Data files
│   ├── department_marketing_mix_data.xlsx
│   ├── faiss_mmm_index.index
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Manya-tech/TacticPlanner.git
   ```
2. Navigate to the project directory:
   ```bash
   cd TacticPlanner
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables in a `.env` file:
   ```env
   MISTRAL_API_KEY=your_api_key_here
   ```
5. Run the application:
   ```bash
   python app.py
   ```

## Usage
1. Open the application in your browser at `http://localhost:8000`.
2. Sign up or log in to access the chat interface.
3. Interact with the AI assistant to get marketing mix recommendations.

## Demo
[Watch the Demo Video](https://example.com/demo-video)

## Live Website
[Visit the Live Website](https://tacticplanner.onrender.com)

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
