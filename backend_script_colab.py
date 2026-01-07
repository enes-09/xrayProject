# =================================================================
# FİNAL X-RAY SUNUCUSU (SENİN MODELLERİNLE)
# =================================================================

# Gerekli işlemler:
!pip install fastapi uvicorn pyngrok python-multipart nest-asyncio

import nest_asyncio
from fastapi import FastAPI, UploadFile, File
from pyngrok import ngrok, conf
import uvicorn
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import torch.nn.functional as F
import os

# --- 1. TOKEN AYARI ---
NGROK_TOKEN = "37nfbRSTCbHSVZ8JS4Q4XPemS6v_86DXzhAAn7XuxPHJZ3VuP"
conf.get_default().auth_token = NGROK_TOKEN

# --- 2. MODEL KONFİGÜRASYONU ---
# Kullanmak istediğin modelin adını buraya yaz:
# Seçenekler: 'swin_t', 'vit_b_16', 'resnet50', 'vgg16', 'chexnet', 'inception_v3'
ACTIVE_MODEL_KEY = 'swin_t'

# Modellerin ve dosya isimlerinin tanımları
# Lütfen dosya isimlerini ("file_path") kendi Colab'a yüklediğin isimlerle güncelle!
MODEL_CONFIGS = {
    'swin_t': {
        'file_path': 'best_swin_t_fold_3.pth',
        'arch': 'swin_t',
        'num_classes': 5
    },
    'vit_b_16': {
        'file_path': 'best_vit_b_16.pth',
        'arch': 'vit_b_16',
        'num_classes': 5
    },
    'resnet50': {
        'file_path': 'best_resnet50.pth',
        'arch': 'resnet50',
        'num_classes': 5
    },
    'vgg16': {
        'file_path': 'best_vgg16.pth',
        'arch': 'vgg16',
        'num_classes': 5
    },
    'chexnet': {
        'file_path': 'best_chexnet.pth',
        'arch': 'densenet121', # CheXNet temelde DenseNet121'dir
        'num_classes': 5
    },
    'inception_v3': {
        'file_path': 'best_inception_v3.pth',
        'arch': 'inception_v3',
        'num_classes': 5
    }
}

# Aktif model ayarlarını çek
CURRENT_CONFIG = MODEL_CONFIGS[ACTIVE_MODEL_KEY]
MODEL_PATH = CURRENT_CONFIG['file_path']
NUM_CLASSES = CURRENT_CONFIG['num_classes']
ARCH_TYPE = CURRENT_CONFIG['arch']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. SINIF İSİMLERİ ---
CLASS_NAMES = {
    0: 'Covid-19',
    1: 'Lung Opacity',
    2: 'Normal',
    3: 'Viral Pneumonia',
    4: 'Tuberculosis'
}

print(f"\n{'!'*60}")
print(f"UYARI: SINIF HARİTASI (CLASS MAPPING) KONTROLÜ")
print(f"{'!'*60}")
print(f"Model şu an şu sıralamayı varsayıyor:")
for k, v in CLASS_NAMES.items():
    print(f"  {k} -> {v}")
print(f"\nEğer sizin eğitim veri setinizdeki klasörlerin (Folder) alfabetik sıralaması")
print(f"bundan farklıysa, tahminler HEP YANLIŞ çıkacaktır!")
print(f"{'!'*60}\n")

# --- 4. MODEL MİMARİSİ VE YARDIMCI FONKSİYONLAR ---
def create_mlp_head(input_dim, num_classes):
    """
    Sınıflandırma katmanı (Classifier Head).
    """
    if input_dim > 1024:
        hidden_1 = 1024
        hidden_2 = 512
    else:
        hidden_1 = 512
        hidden_2 = 256

    return nn.Sequential(
        nn.Linear(input_dim, hidden_1),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_1, hidden_2),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_2, num_classes)
    )

def get_model(arch_type, num_classes):
    print(f">> Mimari hazırlanıyor: {arch_type}")
    model = None
    
    # 1. SWIN TRANSFORMER
    if arch_type == 'swin_t':
        try:
            model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        except:
            model = models.swin_t(pretrained=True)
        in_features = model.head.in_features
        model.head = create_mlp_head(in_features, num_classes)

    # 2. VISION TRANSFORMER (ViT)
    elif arch_type == 'vit_b_16':
        try:
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        except:
            model = models.vit_b_16(pretrained=True)
        in_features = model.heads.head.in_features
        # HATA DUZELTME: Egitimde model.heads direkt degistirilmis (heads.0.weight vs heads.head.0.weight)
        model.heads = create_mlp_head(in_features, num_classes)

    # 3. RESNET (ResNet50)
    elif arch_type == 'resnet50':
        try:
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except:
            model = models.resnet50(pretrained=True)
        in_features = model.fc.in_features
        model.fc = create_mlp_head(in_features, num_classes)

    # 4. VGG (VGG16)
    elif arch_type == 'vgg16':
        try:
            model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        except:
            model = models.vgg16(pretrained=True)
        in_features = model.classifier[6].in_features
        model.classifier[6] = create_mlp_head(in_features, num_classes)

    # 5. DENSENET / CHEXNET
    # CheXNet genellikle DenseNet121 mimarisini kullanır
    elif arch_type == 'densenet121':
        try:
            model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        except:
            model = models.densenet121(pretrained=True)
        in_features = model.classifier.in_features
        model.classifier = create_mlp_head(in_features, num_classes)

    # 6. INCEPTION V3
    elif arch_type == 'inception_v3':
        try:
            # Inception v3 aux_logits=True ile gelir, transform input size 299x299 olmalıdır
            model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        except:
            model = models.inception_v3(pretrained=True)
        
        # Inception'da aux_logits varsa onu da ayarlamak gerekebilir ama 
        # sadece inference (tahmin) yapacagimiz icin ana 'fc' katmanini degistirmek yeterli.
        # Egitim sirasinda aux_logits kullanildiysa model.aux_logits = True kalmalı.
        model.aux_logits = False # Inference'da hata almamak icin kapatalim (State dict'e bagli)
        
        in_features = model.fc.in_features
        model.fc = create_mlp_head(in_features, num_classes)

    else:
        raise ValueError(f"HATA: Tanımlanmamış mimari tipi -> {arch_type}")

    return model.to(device)

# --- 5. MODELİ YÜKLE ---
print(f"\n>> SEÇİLEN MODEL: {ACTIVE_MODEL_KEY} ({ARCH_TYPE})")
print(f">> Dosya Yolu: {MODEL_PATH}")

if os.path.exists(MODEL_PATH):
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f">> Dosya Boyutu: {size_mb:.2f} MB")
else:
    print(f">> UYARI: '{MODEL_PATH}' diskte bulunamadı!")
    print(f">> Mevcut Dosyalar: {os.listdir('.')}")   

print(">> Yükleme başlıyor...")

try:
    model = get_model(ARCH_TYPE, NUM_CLASSES)
    
    # State dict yükle
    state_dict = torch.load(MODEL_PATH, map_location=device)
    
    # GÜÇLENDİRİLMİŞ PREFIX TEMİZLİĞİ:
    # Hem 'module.', hem '1.' hem de iç içe geçmiş 'module.1.' gibi yapıları temizler.
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        # Basında 'module.' veya '1.' olduğu sürece döngüyle kırp
        while name.startswith('module.') or name.startswith('1.'):
            if name.startswith('module.'):
                name = name[7:]
            elif name.startswith('1.'):
                name = name[2:]
        new_state_dict[name] = v
    state_dict = new_state_dict

    # Strict=False ile yükle (Inception aux_logits gibi önemsiz eksikler için)
    model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    print(f">> ✅ {MODEL_PATH} başarıyla yüklendi.")
    
except Exception as e:
    print(f"\n!! HATA: Model yüklenirken bir sorun oluştu.")
    print(f"!! Detay: {e}")

# --- 6. RESİM İŞLEME (Transform) ---
# EĞİTİM İLE EŞLEŞME DÜZELTMESİ:
# Eğitim sırasında muhtemelen resimler önce 224'e küçültülüp, sonra Inception için 299'a çıkarıldı.
# Bu işlem resimde bir miktar "bulanıklık" yaratır. Model bunu öğrendiği için,
# tahmin sırasında da aynı işlemi taklit etmeliyiz.

final_size = 224
if ARCH_TYPE == 'inception_v3':
    final_size = 299 # Inception'ın orijinal girişi
    print(f">> Bilgi: InceptionV3 için özel transform uygulanıyor (224 -> 299 Upsample Simülasyonu)")
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)), # Önce eğitildiği boyuta indir
        transforms.Resize((299, 299)), # Sonra modelin giriş boyutuna çıkar (Upsample effect)
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
else:
    # Diğerleri zaten 224 ile çalışıyor
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# --- 7. SUNUCU (API) ---
app = FastAPI()

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Resmi oku
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        tensor = val_transform(image).unsqueeze(0).to(device)

        # Tahmin
        with torch.no_grad():
            outputs = model(tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        class_id = predicted.item()
        class_name = CLASS_NAMES.get(class_id, f"Bilinmiyor ({class_id})")
        score = confidence.item()

        return {
            "className": class_name,
            "confidence": round(score, 4),
            "source_model": ACTIVE_MODEL_KEY,
            "message": f"Tespit: {class_name}"
        }
    except Exception as e:
        print(f"HATA: {e}")
        return {"className": "Hata", "confidence": 0.0, "message": str(e)}

# --- 8. BAŞLAT ---
ngrok.kill()
ngrok_tunnel = ngrok.connect(8000)
print('\n' + '='*60)
print(f'🚀 LİNKİNİZ: {ngrok_tunnel.public_url}/predict')
print('='*60 + '\n')

nest_asyncio.apply()
config = uvicorn.Config(app, port=8000)
server = uvicorn.Server(config)
await server.serve()
