# RAG_Langchain
RAG

**Langchain RAG**

                   ┌────────────────────────────────────────┐
                   │   Fetch YouTube Transcript (Hindi)     │
                   └────────────────────────────────────────┘
                                   │
                                   ▼
                 ┌────────────────────────────────────┐
                 │ Split Transcript into Overlapping  │
                 │ Text Chunks using RecursiveSplitter│
                 └────────────────────────────────────┘
                                   │
                                   ▼
            ┌──────────────────────────────────────────────┐
            │ Embed Chunks with Sentence Transformers      │
            │ (all-MiniLM-L6-v2) and store in FAISS index  │
            └──────────────────────────────────────────────┘
                                   │
                                   ▼
        ┌──────────────────────────────────────────────┐
        │ Retrieve Top-k Relevant Chunks for Question  │
        └──────────────────────────────────────────────┘
                                   │
                                   ▼
     ┌────────────────────────────────────────────────────────┐
     │ Construct Prompt using Retrieved Context and Question  │
     └────────────────────────────────────────────────────────┘
                                   │
                                   ▼
           ┌────────────────────────────────────────┐
           │ Run Prompt through Falcon-RW CPU Model │
           └────────────────────────────────────────┘
                                   │
                                   ▼
             ┌────────────────────────────────────┐
             │      Parse and Print the Answer     │
             └────────────────────────────────────┘
