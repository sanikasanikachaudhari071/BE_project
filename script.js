// Simple Deepfake Detection Frontend

// Get elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const loading = document.getElementById('loading');
const imagePreview = document.getElementById('imagePreview');
const resultArea = document.getElementById('resultArea');

let selectedFile = null;

// Click upload area to select file
uploadArea.onclick = () => fileInput.click();

// File selected
fileInput.onchange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    selectedFile = file;
    analyzeBtn.disabled = false;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.innerHTML = `<img src="${e.target.result}" class="preview-image">`;
    };
    reader.readAsDataURL(file);
};

// Analyze button clicked
analyzeBtn.onclick = () => {
    if (!selectedFile) return;
    
    // Show loading
    loading.style.display = 'block';
    resultArea.style.display = 'none';
    
    // Show error after 2 seconds
    setTimeout(() => {
        loading.style.display = 'none';
        resultArea.innerHTML = `
            <div style="background: #ff6b6b; color: white; padding: 20px; border-radius: 10px;">
                <h3>❌ Backend Not Connected</h3>
                <p>This is a frontend demo only.</p>
                <p>No real analysis is performed.</p>
            </div>
        `;
        resultArea.style.display = 'block';
    }, 2000);
};

// Clear button clicked
clearBtn.onclick = () => {
    selectedFile = null;
    fileInput.value = '';
    analyzeBtn.disabled = true;
    imagePreview.innerHTML = '';
    resultArea.style.display = 'none';
    loading.style.display = 'none';
};

// Drag and drop
uploadArea.ondragover = (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '#667eea';
};

uploadArea.ondragleave = () => {
    uploadArea.style.borderColor = '#ddd';
};

uploadArea.ondrop = (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '#ddd';
    const file = e.dataTransfer.files[0];
    if (file) {
        fileInput.files = e.dataTransfer.files;
        fileInput.onchange({ target: { files: [file] } });
    }
};

console.log('Frontend loaded - Demo only');
