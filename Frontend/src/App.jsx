import { useState } from 'react';
import axios from 'axios';
import './App.css';

// Kullanılabilir modeller
const AVAILABLE_MODELS = [
    { key: 'all', name: 'Tümü (Tüm Modeller)' },
    { key: 'swin_t', name: 'Swin Transformer' },
    { key: 'vit_b_16', name: 'ViT (Vision Transformer)' },
    { key: 'resnet50', name: 'ResNet50' },
    { key: 'vgg16', name: 'VGG16' },
    { key: 'chexnet', name: 'CheXNet (DenseNet)' },
    { key: 'inception_v3', name: 'Inception V3' }
];

// Sadece gerçek modeller (all hariç)
const ACTUAL_MODELS = AVAILABLE_MODELS.filter(m => m.key !== 'all');

function App() {
    // Çoklu dosya desteği
    const [files, setFiles] = useState([]);
    const [previews, setPreviews] = useState([]);
    const [selectedIndex, setSelectedIndex] = useState(null);

    // Sonuç state'leri
    const [result, setResult] = useState(null);          // Tek model sonucu
    const [multiResults, setMultiResults] = useState([]); // Çoklu model sonuçları
    const [loading, setLoading] = useState(false);
    const [loadingProgress, setLoadingProgress] = useState({ current: 0, total: 0 });
    const [error, setError] = useState(null);
    const [selectedModel, setSelectedModel] = useState('swin_t');

    const handleFilesChange = (e) => {
        const selectedFiles = Array.from(e.target.files);
        if (selectedFiles.length > 0) {
            const newFiles = [...files, ...selectedFiles];
            const newPreviews = [...previews, ...selectedFiles.map(file => URL.createObjectURL(file))];

            setFiles(newFiles);
            setPreviews(newPreviews);

            if (selectedIndex === null) {
                setSelectedIndex(0);
            }

            setResult(null);
            setMultiResults([]);
            setError(null);
        }
    };

    const handleSelectImage = (index) => {
        setSelectedIndex(index);
        setResult(null);
        setMultiResults([]);
        setError(null);
    };

    const handleRemoveImage = (index, e) => {
        e.stopPropagation();

        const newFiles = files.filter((_, i) => i !== index);
        const newPreviews = previews.filter((_, i) => i !== index);

        URL.revokeObjectURL(previews[index]);

        setFiles(newFiles);
        setPreviews(newPreviews);

        if (newFiles.length === 0) {
            setSelectedIndex(null);
            setResult(null);
            setMultiResults([]);
        } else if (selectedIndex >= newFiles.length) {
            setSelectedIndex(newFiles.length - 1);
        } else if (selectedIndex === index) {
            setSelectedIndex(Math.max(0, index - 1));
            setResult(null);
            setMultiResults([]);
        }
    };

    const handleClearAll = () => {
        previews.forEach(url => URL.revokeObjectURL(url));

        setFiles([]);
        setPreviews([]);
        setSelectedIndex(null);
        setResult(null);
        setMultiResults([]);
        setError(null);
    };

    const handleAnalyze = async () => {
        if (selectedIndex === null || !files[selectedIndex]) return;

        setLoading(true);
        setError(null);
        setResult(null);
        setMultiResults([]);

        try {
            if (selectedModel === 'all') {
                // Tüm modeller için paralel analiz
                setLoadingProgress({ current: 0, total: ACTUAL_MODELS.length });

                const results = [];

                // Paralel istekler gönder
                const promises = ACTUAL_MODELS.map(async (model, index) => {
                    const formData = new FormData();
                    formData.append('image', files[selectedIndex]);
                    formData.append('modelName', model.key);

                    try {
                        const response = await axios.post('http://localhost:8080/api/xray/upload', formData, {
                            headers: { 'Content-Type': 'multipart/form-data' },
                        });

                        setLoadingProgress(prev => ({ ...prev, current: prev.current + 1 }));

                        return {
                            modelKey: model.key,
                            modelName: model.name,
                            ...response.data,
                            success: true
                        };
                    } catch (err) {
                        setLoadingProgress(prev => ({ ...prev, current: prev.current + 1 }));
                        return {
                            modelKey: model.key,
                            modelName: model.name,
                            className: 'Hata',
                            confidence: 0,
                            message: 'Analiz başarısız',
                            success: false
                        };
                    }
                });

                const allResults = await Promise.all(promises);

                // Güven skoruna göre sırala (en yüksek önce)
                allResults.sort((a, b) => b.confidence - a.confidence);

                setMultiResults(allResults);
            } else {
                // Tek model için normal analiz
                const formData = new FormData();
                formData.append('image', files[selectedIndex]);
                formData.append('modelName', selectedModel);

                const response = await axios.post('http://localhost:8080/api/xray/upload', formData, {
                    headers: { 'Content-Type': 'multipart/form-data' },
                });

                setResult(response.data);
            }
        } catch (err) {
            console.error(err);
            setError("Bağlantı hatası! Backend çalışıyor mu?");
        } finally {
            setLoading(false);
            setLoadingProgress({ current: 0, total: 0 });
        }
    };

    // En iyi sonucu bul (multiResults için)
    const getBestResult = () => {
        if (multiResults.length === 0) return null;
        return multiResults[0]; // Zaten sıralı, ilk eleman en yüksek güven skorlu
    };

    return (
        <div className="app-container">
            {/* Sol Panel: Kontroller */}
            <div className="sidebar">
                <div className="brand-header">
                    <h1 className="brand-title">Xray Görüntü Analizi<br />Berkay Özer, Enes Kulpu</h1>
                </div>

                <div className="upload-section">
                    <p className="instructions">Analiz için X-Ray görüntüleri seçin:</p>

                    {/* Model Seçimi */}
                    <div className="model-selector">
                        <label htmlFor="modelSelect" className="model-label">Model Seçin:</label>
                        <select
                            id="modelSelect"
                            className="model-dropdown"
                            value={selectedModel}
                            onChange={(e) => setSelectedModel(e.target.value)}
                        >
                            {AVAILABLE_MODELS.map((model) => (
                                <option key={model.key} value={model.key}>
                                    {model.name}
                                </option>
                            ))}
                        </select>
                    </div>

                    {/* Çoklu Dosya Seçimi */}
                    <input
                        type="file"
                        accept="image/*"
                        multiple
                        onChange={handleFilesChange}
                        id="fileInput"
                    />
                    <label htmlFor="fileInput" className="file-label">
                        <span>Dosya Seç veya Sürükle</span>
                    </label>

                    {/* Galeri */}
                    {previews.length > 0 && (
                        <div className="gallery-section">
                            <div className="gallery-header">
                                <span className="gallery-count">{previews.length} görüntü</span>
                                <button className="clear-all-btn" onClick={handleClearAll}>
                                    Tümünü Temizle
                                </button>
                            </div>
                            <div className="thumbnail-gallery">
                                {previews.map((preview, index) => (
                                    <div
                                        key={index}
                                        className={`thumbnail-item ${selectedIndex === index ? 'selected' : ''}`}
                                        onClick={() => handleSelectImage(index)}
                                        title={files[index]?.name || `Görüntü ${index + 1}`}
                                    >
                                        <img src={preview} alt={files[index]?.name || `Görüntü ${index + 1}`} />
                                        <button
                                            className="remove-btn"
                                            onClick={(e) => handleRemoveImage(index, e)}
                                            title="Sil"
                                        >
                                            ×
                                        </button>
                                        <span className="thumbnail-number">{index + 1}</span>
                                    </div>
                                ))}
                            </div>
                            {selectedIndex !== null && files[selectedIndex] && (
                                <div className="selected-file-name">
                                    {files[selectedIndex].name}
                                </div>
                            )}
                        </div>
                    )}

                </div>

                {/* Sabit Alt Kısım - Error ve Buton */}
                <div className="sidebar-footer">
                    {error && <div className="error-msg">{error}</div>}
                    <button
                        onClick={handleAnalyze}
                        disabled={selectedIndex === null || loading}
                        className="analyze-btn"
                    >
                        {loading
                            ? (selectedModel === 'all'
                                ? `Analiz ediliyor... (${loadingProgress.current}/${loadingProgress.total})`
                                : "Analiz ediliyor...")
                            : (selectedModel === 'all'
                                ? "Tüm Modellerle Analiz Et"
                                : "Seçili Görüntüyü Analiz Et")
                        }
                    </button>
                </div>
            </div>

            {/* Sağ Panel: Görüntü ve Sonuç */}
            <div className="main-content">
                {selectedIndex === null ? (
                    <div className="empty-state">
                        <h3>Görüntü Bekleniyor</h3>
                        <p>Analiz için bir veya daha fazla X-Ray görüntüsü yükleyin.</p>
                    </div>
                ) : (
                    <div className="result-card">
                        <div className="image-container">
                            <img src={previews[selectedIndex]} alt="Analiz" className="preview-img" />
                        </div>

                        {/* Tek Model Sonucu */}
                        {result && (
                            <div className="result-details">
                                <span className={`result-badge ${result.className === 'Normal' ? 'badge-success' : 'badge-danger'
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

                        {/* Çoklu Model Sonuçları */}
                        {multiResults.length > 0 && (
                            <div className="multi-results">
                                <h3 className="multi-results-title">Tüm Model Sonuçları</h3>

                                {/* Sonuç Tablosu */}
                                <div className="results-table-container">
                                    <table className="results-table">
                                        <thead>
                                            <tr>
                                                <th>Model</th>
                                                <th>Sonuç</th>
                                                <th>Güven Skoru</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {multiResults.map((res, idx) => (
                                                <tr key={res.modelKey} className={idx === 0 ? 'best-row' : ''}>
                                                    <td>{res.modelName}</td>
                                                    <td>
                                                        <span className={`table-badge ${res.className === 'Normal' ? 'badge-success' : 'badge-danger'
                                                            }`}>
                                                            {res.className}
                                                        </span>
                                                    </td>
                                                    <td className="confidence-cell">
                                                        %{(res.confidence * 100).toFixed(2)}
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}

export default App;