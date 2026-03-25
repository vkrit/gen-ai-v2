# Exercise 1: Visualize Thai Medical Embeddings with PCA
# Install: pip install sentence-transformers scikit-learn matplotlib

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'TH Sarabun New'  # Thai font support

model = SentenceTransformer("BAAI/bge-m3")

# --- Thai medical sentences grouped by topic ---
sentences = [
    # Drug side effects
    "พาราเซตามอลอาจทำให้ตับอักเสบหากรับประทานเกินขนาด",
    "ผลข้างเคียงของยาแอสไพรินได้แก่ระคายเคืองกระเพาะอาหาร",
    "ยาไอบูโพรเฟนอาจทำให้เกิดแผลในกระเพาะอาหาร",
    # Vaccination
    "วัคซีนไข้หวัดใหญ่ควรฉีดปีละครั้ง",
    "วัคซีน COVID-19 ช่วยลดความรุนแรงของโรค",
    "เด็กควรได้รับวัคซีนตามตารางที่กระทรวงสาธารณสุขกำหนด",
    # PDPA / data privacy
    "การละเมิดข้อมูลส่วนบุคคลมีโทษจำคุกไม่เกินหนึ่งปี",
    "โรงพยาบาลต้องขอความยินยอมก่อนเปิดเผยข้อมูลผู้ป่วย",
    "พระราชบัญญัติคุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562",
]

labels = [
    "Drug SE 1", "Drug SE 2", "Drug SE 3",
    "Vaccine 1", "Vaccine 2", "Vaccine 3",
    "PDPA 1",    "PDPA 2",    "PDPA 3",
]

colors = ["#e74c3c"]*3 + ["#2ecc71"]*3 + ["#3498db"]*3

# --- Embed and reduce to 2D ---
embeddings = model.encode(sentences, normalize_embeddings=True)
pca = PCA(n_components=2)
coords = pca.fit_transform(embeddings)

# --- Plot ---
plt.figure(figsize=(8, 6))
for i, (x, y) in enumerate(coords):
    plt.scatter(x, y, color=colors[i], s=120, zorder=3)
    plt.annotate(labels[i], (x, y), textcoords="offset points",
                 xytext=(6, 4), fontsize=9)

plt.title("Thai Medical Sentences — PCA Projection of Embeddings")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.tight_layout()
plt.savefig("embedding_clusters.png", dpi=150)
plt.show()
