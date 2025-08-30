# 📚 Semantic Book Recommender  

A semantic book recommendation system built with **LangChain**, **OpenAI embeddings**, **Hugging Face Transformers**, and a **Gradio UI**.  

The project explores **data cleaning, feature engineering, vector search, zero-shot classification, and emotion analysis** to create an interactive dashboard for book discovery.  

---

## 🚀 Features  

- **Data Cleaning & Feature Engineering**  
  - Handling missing values and engineered features (book age, missing description flags).  
  - Correlation analysis & visualization with Seaborn/Matplotlib.  
  - Description length analysis for NLP readiness.  

- **Vector Search with LangChain**  
  - Convert book descriptions + ISBNs into embeddings using **OpenAIEmbeddings**.  
  - Store and query using **ChromaDB**.  
  - Semantic similarity search: e.g., *“A book to teach children about nature”*.  

- **Text Classification**  
  - Zero-shot classification (Hugging Face `facebook/bart-large-mnli`) to map categories → Fiction, Nonfiction, Children’s Fiction, Children’s Nonfiction.  
  - Automated category filling for missing values.  

- **Emotion Analysis**  
  - Hugging Face emotion classification model.  
  - Scores for **anger, disgust, fear, joy, sadness, surprise, neutral**.  
  - Merged back into the dataset for tone-based recommendations.  

- **Interactive Gradio Dashboard**  
  - User inputs query + category + tone.  
  - Displays recommendations with cover thumbnails, titles, authors, and descriptions.  
  - Supports filtering by **semantic similarity**, **category**, and **tone** (Happy, Sad, Angry, Suspenseful, Surprising).  

---

## 🛠️ Tech Stack  

- **Python**: Data processing & orchestration  
- **Pandas / NumPy**: Data manipulation  
- **Seaborn / Matplotlib**: Visualization  
- **LangChain (community, openai, chroma)**: Vector search pipeline  
- **OpenAI API**: Embeddings  
- **Hugging Face Transformers**: Zero-shot classification & emotion models  
- **Gradio**: Dashboard UI  
- **Docker**: Containerization for deployment  

---

## 📦 Installation  

1. Clone the repo:  
   ```bash
   git clone https://github.com/your-username/semantic-book-recommender.git
   cd semantic-book-recommender
   ```

2. Create a virtual environment:  
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS/Linux  
   .venv\Scripts\activate      # Windows
   ```

3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

4. Add your **OpenAI API key** in a `.env` file:  
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

---

## ▶️ Usage  

### Run the Dashboard  
```bash
python gradio-UI.py
```

The app will launch locally at: [http://localhost:7860] 

### Example Query  
- Query: *“A story about forgiveness”*  
- Category: *Fiction*  
- Tone: *Sad*  

The system returns the top 16 most relevant books.  

---

## 📊 Workflow Overview  

1. **Data Cleaning**  
   - Visualize missingness with heatmaps.  
   - Feature engineering (age of book, missing description flags).  
   - Correlation analysis using **Spearman**.  

2. **Vector Search**  
   - Join ISBN with description.  
   - Create embeddings (OpenAI).  
   - Store & query in ChromaDB.  

3. **Zero-Shot Classification**  
   - Hugging Face BART model predicts Fiction vs. Nonfiction (and Children’s sub-categories).  
   - Fill missing categories automatically.  

4. **Emotion Classification**  
   - Apply pre-trained emotion classifier to descriptions.  
   - Aggregate sentence-level scores → max emotion per book.  

5. **Gradio Dashboard**  
   - Query semantic recommendations.  
   - Filter by category/tone.  
   - Display results with images + captions.  

---

## 🐳 Docker  

1. Build image:  
   ```bash
   docker build -t semantic-book-recommender .
   ```

2. Run container:  
   ```bash
   docker run -p 7860:7860 semantic-book-recommender
   ```

The app will be available at `http://localhost:7860`.  

---

## 📂 Project Structure  

```
📁 BookRecommender/
 ├── 📄 requirements.txt
 ├── 📄 gradio-UI.py          # Main Gradio dashboard
 ├── 📄 data_cleaning.ipynb   # Data cleaning & feature engineering
 ├── 📄 vector_search.ipynb   # LangChain + Chroma vector DB
 ├── 📄 text-classification.ipynb
 ├── 📄 emotion-analysis.ipynb
 ├── 📄 tagged_description.txt
 ├── 📄 books_with_emotions.csv
 └── 📄 README.md
```
