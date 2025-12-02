🔐 Privacy-Preserving Similarity Search

This project implements a secure similarity search framework that enables users to retrieve similar data records without revealing sensitive information to the cloud server. It combines Paillier Homomorphic Encryption and Vantage Point (VP) Trees to perform efficient similarity searches directly on encrypted data.

📌 Key Features

End-to-end encryption: Both dataset and user queries remain encrypted during storage and processing.

Homomorphic computations: Cloud performs similarity calculations without decrypting data.

VP Tree-based indexing: Efficient hierarchical data partitioning for high-dimensional similarity search.

User-secure workflow: Only the user can decrypt the final results.

Streamlit UI: Simple and interactive interface for uploading data, encrypting queries, and retrieving encrypted results.

🧠 Technologies Used

Paillier Homomorphic Encryption

Vantage Point Tree (VP-Tree)

Python

NumPy / SciPy

Streamlit (for web interface)

🚀 How It Works

User encrypts dataset and query using Paillier HE.

Encrypted data is uploaded to the cloud server.

Cloud constructs a VP Tree on encrypted data for efficient search.

Similarity search is performed homomorphically using encrypted distance computation.

Cloud returns encrypted results to the user.

User decrypts results locally, ensuring complete data confidentiality.

📌 Applications

Healthcare: Encrypted patient record comparison

Finance: Fraud detection without exposing transaction data

E-commerce: Privacy-preserving recommendation systems

Biometrics: Secure encrypted face/fingerprint matching
