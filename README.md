# GIKAtlas – AI-Powered Intelligent Campus Navigation System

GIKAtlas is an artificial intelligence–based campus navigation system designed for Ghulam Ishaq Khan Institute of Engineering Sciences and Technology (GIKI). The system models the campus as a weighted graph, applies classical AI search algorithms for route planning, and integrates machine learning models to predict Estimated Time of Arrival (ETA) and route reliability.



##  Features
- Graph-based modeling of the GIKI campus
- Implementation of classical AI search algorithms:
  - Breadth First Search (BFS)
  - Depth First Search (DFS)
  - Dijkstra’s Algorithm
  - Greedy Best-First Search (GBFS)
  - A* Search Algorithm
- ETA prediction using Linear Regression
- Route classification (Short / Medium / Long) using Logistic Regression
- Performance evaluation using MAE, R² Score, and Accuracy
- Interactive web interface using Streamlit
- Visual route display using Plotly



##  System Architecture
The system consists of three main components:
1. **Graph-Based Navigation Engine** – Computes optimal routes using AI algorithms  
2. **Machine Learning Module** – Predicts ETA and route reliability  
3. **User Interface** – Streamlit-based interactive web application  


##  Machine Learning Models
### ETA Prediction
- Model: Linear Regression
- Features: Distance, number of turns, selected algorithm
- Output: Estimated Time of Arrival (minutes)

### Route Classification
- Model: Logistic Regression
- Classes: Short, Medium, Long



##  Performance Metrics
- **Mean Absolute Error (MAE)** for ETA prediction
- **R² Score** for regression performance
- **Accuracy** for route classification



##  Tools & Technologies
- Python
- Streamlit
- Plotly
- Scikit-learn
- Pandas



##  How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
