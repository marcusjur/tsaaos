from sentence_transformers import SentenceTransformer

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

# Initialize the model
model = SentenceTransformer('intfloat/multilingual-e5-large')

# load one sentence per line
with open('sentences.txt', 'r', encoding='utf-8') as file:
    input_texts = [line.strip() for line in file.readlines()]

# Encode the sentences to get the embeddings
embeddings = model.encode(input_texts, normalize_embeddings=True)

# Use PCA to reduce the embeddings to 3 dimensions for visualization
pca = PCA(n_components=3)
embeddings_3d = pca.fit_transform(embeddings)

# Use TSNE to reduce the embeddings to 3 dimensions for visualization
# tsne = TSNE(n_components=3, perplexity=1, learning_rate=200, n_iter=1000, random_state=42)
# embeddings_3d = tsne.fit_transform(embeddings)

plt.style.use('dark_background')

# Create a 3D plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Scatter points
sc = ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2])

# Initialize the annotation object and make it invisible initially
annot = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"))
annot.set_visible(False)


# update the annotation
def update_annot(ind):
    # Get the data point index of the hover
    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}".format(" ".join([input_texts[n] for n in ind["ind"]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.4)


# control the hover behavior
def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()


# Connect the hover function to the figure canvas
fig.canvas.mpl_connect("motion_notify_event", hover)

# Set labels
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')

# Show the plot
plt.show()
