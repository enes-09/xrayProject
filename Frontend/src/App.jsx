import { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile) {
            setFile(selectedFile);
            setPreview(URL.createObjectURL(selectedFile));
            setResult(null);
            setError(null);
        }
    };

    const handleAnalyze = async () => {
        if (!file) return;

        setLoading(true);
        setError(null);
        setResult(null);

        const formData = new FormData();
        formData.append('image', file);

        try {
            // Backend adresi (localhost:8080)
            const response = await axios.post('http://localhost:8080/api/xray/upload', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });

            setResult(response.data);
        } catch (err) {
            console.error(err);
            setError("Bağlantı hatası! Backend çalışıyor mu?");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="container">
            <div className="card">
                <h1>🩻 X-Ray AI Analiz</h1>
                <p>Lütfen analiz edilecek röntgen görüntüsünü yükleyin.</p>

                <div className="upload-area">
                    <input type="file" accept="image/*" onChange={handleFileChange} id="fileInput" />
                    <label htmlFor="fileInput" className="file-label">
                        {file ? file.name : "Dosya Seç veya Sürükle"}
                    </label>
                </div>

                {preview && (
                    <div className="preview-box">
                        <img src={preview} alt="Önizleme" className="preview-img" />
                    </div>
                )}

                <button onClick={handleAnalyze} disabled={!file || loading} className="analyze-btn">
                    {loading ? "AI İnceliyor..." : "Analiz Et"}
                </button>

                {error && <div className="error-msg">{error}</div>}

                {result && (
                    <div className={`result-box ${result.className === 'Pneumonia' ? 'danger' : 'success'}`}>
                        <h2>Sonuç: {result.className === 'Pneumonia' ? 'Zatürre Riski' : 'Normal'}</h2>
                        <p>Güven Skoru: <strong>%{ (result.confidence * 100).toFixed(2) }</strong></p>
                        <small>{result.message}</small>
                    </div>
                )}
            </div>
        </div>
    );
}

export default App;