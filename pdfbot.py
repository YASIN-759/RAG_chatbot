import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

class PDFQAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Q&A with FAISS Embeddings")
        self.root.geometry("700x600")
        
        # Embedding model
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        self.db_folder = None
        self.vectorstore = None
        
        # UI Elements
        self.load_btn = tk.Button(root, text="Load PDF", command=self.load_pdf)
        self.load_btn.pack(pady=10)
        
        self.question_label = tk.Label(root, text="Ask your question:")
        self.question_label.pack()
        
        self.question_entry = tk.Entry(root, width=80)
        self.question_entry.pack(pady=5)
        
        self.ask_btn = tk.Button(root, text="Ask", command=self.ask_question, state=tk.DISABLED)
        self.ask_btn.pack(pady=10)
        
        self.result_box = scrolledtext.ScrolledText(root, width=80, height=25)
        self.result_box.pack(pady=10)
    
    def load_pdf(self):
        # Ask user to select PDF file
        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if not file_path:
            return
        
        try:
            # Load PDF
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Split to chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_documents(documents)
            
            # Create FAISS vectorstore
            self.vectorstore = FAISS.from_documents(chunks, self.embedding_model)
            
            # Save FAISS index in a folder near the PDF file
            base_folder = os.path.dirname(file_path)
            self.db_folder = os.path.join(base_folder, "faiss_index")
            self.vectorstore.save_local(self.db_folder)
            
            messagebox.showinfo("Success", f"PDF loaded and embedded!\nFAISS index saved at:\n{self.db_folder}")
            self.ask_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load PDF:\n{str(e)}")
    
    def ask_question(self):
        query = self.question_entry.get().strip()
        if not query:
            messagebox.showwarning("Input needed", "Please enter a question.")
            return
        
        if self.vectorstore is None:
            # Try to load vectorstore if possible
            if self.db_folder and os.path.exists(self.db_folder):
                try:
                    self.vectorstore = FAISS.load_local(self.db_folder, self.embedding_model, allow_dangerous_deserialization=True)
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load FAISS index:\n{str(e)}")
                    return
            else:
                messagebox.showwarning("Load PDF first", "Please load a PDF first.")
                return
        
        # Search top 3 relevant docs
        try:
            results = self.vectorstore.similarity_search(query, k=3)
        except Exception as e:
            messagebox.showerror("Search Error", str(e))
            return
        
        # Show results
        self.result_box.delete(1.0, tk.END)
        if not results:
            self.result_box.insert(tk.END, "No relevant documents found.")
            return
        
        for i, doc in enumerate(results, 1):
            self.result_box.insert(tk.END, f"Result {i}:\n{doc.page_content}\n\n")
        

if __name__ == "__main__":
    root = tk.Tk()
    app = PDFQAApp(root)
    root.mainloop()
