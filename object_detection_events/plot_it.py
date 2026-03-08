import matplotlib.pyplot as plt 
from collections import Counter
                                                                                                                                                                                            
data = [
    "suitcase", "suitcase", "suitcase", "suitcase", "suitcase",                                                                                                                             
    "bird",     
    "bed", "bed", "bed", "bed", "bed", "bed", "bed", "bed", "bed",
    "bed", "bed", "bed", "bed", "bed", "bed", "bed", "bed", "bed",
    "suitcase", "suitcase",
    "sports ball", "sports ball", "sports ball", "sports ball",
    "sports ball", "sports ball", "sports ball",
]

counts = Counter(data)
labels = list(counts.keys())
values = list(counts.values())

plt.bar(labels, values)
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Object Detection Class Frequency")
plt.tight_layout()
plt.show()
