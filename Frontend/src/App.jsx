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
        <div className="app-container">
            {/* Sol Panel: Kontroller */}
            <div className="sidebar">
                <div className="brand-header">
                    <h1 className="brand-title">Xray Görüntü Analizi<br />Berkay Özer, Enes Kulpu</h1>
                </div>

                <div className="upload-section">
                    <p className="instructions">Analiz için bir X-Ray görüntüsü seçin:</p>

                    <input type="file" accept="image/*" onChange={handleFileChange} id="fileInput" />
                    <label htmlFor="fileInput" className="file-label">
                        {file ? (
                            <span>Görsel Seçildi</span>
                        ) : (
                            <span>📁 Dosya Seç veya Sürükle</span>
                        )}
                    </label>
                    {file && <div className="file-name">{file.name}</div>}

                    <div style={{ flex: 1 }}></div> {/* Spacer */}

                    {error && <div className="error-msg">{error}</div>}

                    <button
                        onClick={handleAnalyze}
                        disabled={!file || loading}
                        className="analyze-btn"
                    >
                        {loading ? "Analiz ediliyor" : "Analiz Et"}
                    </button>
                </div>
            </div>

            {/* Sağ Panel: Görüntü ve Sonuç */}
            <div className="main-content">
                {!preview ? (
                    <div className="empty-state">

                        <h3>Görüntü Bekleniyor</h3>
                        <p>Analiz sonuçları burada görüntülenecektir.</p>
                    </div>
                ) : (
                    <div className="result-card">
                        <div className="image-container">
                            <img src={preview} alt="Analiz" className="preview-img" />
                        </div>

                        {result && (
                            <div className="result-details">
                                <span className={`result-badge ${result.className === 'Pneumonia' || result.className.toLowerCase().includes('hata')
                                    ? 'badge-danger'
                                    : 'badge-success'
                                    }`}>
                                    {result.className}
                                </span>
                                <p className="confidence-text">
                                    Güven Skoru: <strong>%{(result.confidence * 100).toFixed(2)}</strong>
                                </p>
                                {result.source_model && (
                                    <p className="model-info">
                                        Kullanılan Model: <span className="model-name">{result.source_model}</span>
                                    </p>
                                )}
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div >
    );
}

export default App;