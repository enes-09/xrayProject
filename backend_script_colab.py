# =================================================================
# FİNAL X-RAY SUNUCUSU (DİNAMİK MODEL SEÇİMİ)
# =================================================================

# Gerekli işlemler:
!pip install fastapi uvicorn pyngrok python-multipart nest-asyncio

import nest_asyncio
from fastapi import FastAPI, UploadFile, File, Form
from pyngrok import ngrok, conf
import uvicorn
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import torch.nn.functional as F
import os
from typing import Optional

# --- 1. TOKEN AYARI ---
NGROK_TOKEN = "37nfbRSTCbHSVZ8JS4Q4XPemS6v_86DXzhAAn7XuxPHJZ3VuP"
conf.get_default().auth_token = NGROK_TOKEN

# --- 2. MODEL KONFİGÜRASYONU ---
# Varsayılan model (ilk yüklenecek)
DEFAULT_MODEL_KEY = 'swin_t'

# Modellerin ve dosya isimlerinin tanımları
MODEL_CONFIGS = {
    'swin_t': {
        'file_path': 'best_swin_t_fold_2.pth',
        'arch': 'swin_t',
        'num_classes': 5
    },
    'vit_b_16': {
        'file_path': 'best_vit_b_16_fold_3.pth',
        'arch': 'vit_b_16',
        'num_classes': 5
    },
    'resnet50': {
        'file_path': 'best_resnet50_fold_4.pth',
        'arch': 'resnet50',
        'num_classes': 5
    },
    'vgg16': {
        'file_path': 'best_vgg16_fold_2.pth',
        'arch': 'vgg16',
        'num_classes': 5
    },
    'chexnet': {
        'file_path': 'best_chexnet_fold_1.pth',
        'arch': 'densenet121',
        'num_classes': 5
    },
    'inception_v3': {
        'file_path': 'best_inception_v3_fold_1.pth',
        'arch': 'inception_v3',
        'num_classes': 5
    }
}

# --- GLOBAL DEĞİŞKENLER (Dinamik Model Yönetimi) ---
current_model = None
current_model_key = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. SINIF İSİMLERİ ---
CLASS_NAMES = {
    0: 'Covid-19',
    1: 'Lung Opacity',
    2: 'Normal',
    3: 'Viral Pneumonia',
    4: 'Tuberculosis'
}

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
            base_model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        except:
            base_model = models.inception_v3(pretrained=True)

        base_model.aux_logits = False
        in_features = base_model.fc.in_features
        base_model.fc = create_mlp_head(in_features, num_classes)

        # Eğitimdeki yapının AYNISI (Upsample + Model)
        model = nn.Sequential(
            nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False),
            base_model
        )

    else:
        raise ValueError(f"HATA: Tanımlanmamış mimari tipi -> {arch_type}")

    return model.to(device)

def load_model(model_key: str):
    """
    Belirtilen modeli yükler ve global değişkenleri günceller.
    Eğer model zaten yüklüyse tekrar yüklemez.
    """
    global current_model, current_model_key

    # Aynı model zaten yüklüyse tekrar yükleme
    if current_model_key == model_key and current_model is not None:
        print(f">> Model zaten yüklü: {model_key}")
        return current_model

    # Model konfigürasyonunu al
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"HATA: Bilinmeyen model anahtarı -> {model_key}")

    config = MODEL_CONFIGS[model_key]
    model_path = config['file_path']
    arch_type = config['arch']
    num_classes = config['num_classes']

    print(f"\n>> MODEL DEĞİŞTİRİLİYOR: {current_model_key} -> {model_key}")
    print(f">> Dosya Yolu: {model_path}")

    # Eski modeli temizle (GPU belleği için)
    if current_model is not None:
        del current_model
        torch.cuda.empty_cache()
        print(">> Eski model bellekten temizlendi.")

    # Yeni modeli oluştur
    model = get_model(arch_type, num_classes)

    # State dict yükle
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)

        # 'module.' temizliği (DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k
            if name.startswith('module.'):
                name = name[7:]
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=True)
        print(f">> ✅ {model_path} başarıyla yüklendi.")
    else:
        print(f">> ⚠️ UYARI: {model_path} bulunamadı!")

    model.eval()

    # Global değişkenleri güncelle
    current_model = model
    current_model_key = model_key

    return model

# --- 5. BAŞLANGIÇTA VARSAYILAN MODELİ YÜKLE ---
print(f"\n>> VARSAYILAN MODEL YÜKLENİYOR: {DEFAULT_MODEL_KEY}")
try:
    load_model(DEFAULT_MODEL_KEY)
except Exception as e:
    print(f"!! HATA: Varsayılan model yüklenemedi -> {e}")

# --- 6. RESİM İŞLEME ---
input_size = 224

val_transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 7. SUNUCU (API) ---
app = FastAPI()

@app.get("/models")
async def get_available_models():
    """
    Kullanılabilir modellerin listesini döndürür.
    Frontend bu endpoint'i çağırarak dropdown'u doldurabilir.
    """
    return {
        "models": list(MODEL_CONFIGS.keys()),
        "current_model": current_model_key
    }

@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None)
):
    """
    X-Ray görüntüsünü analiz eder.
    - file: Yüklenecek görüntü dosyası
    - model_name: Kullanılacak modelin anahtarı (opsiyonel, varsayılan: mevcut model)
    """
    global current_model, current_model_key

    try:
        # Model seçimi (gönderilmediyse mevcut modeli kullan)
        requested_model = model_name if model_name else current_model_key

        # Geçerli model mi kontrol et
        if requested_model not in MODEL_CONFIGS:
            return {
                "className": "Hata",
                "confidence": 0.0,
                "source_model": None,
                "message": f"Geçersiz model: {requested_model}. Geçerli modeller: {list(MODEL_CONFIGS.keys())}"
            }

        # Gerekirse modeli değiştir
        if requested_model != current_model_key:
            try:
                load_model(requested_model)
            except Exception as e:
                return {
                    "className": "Hata",
                    "confidence": 0.0,
                    "source_model": current_model_key,
                    "message": f"Model yüklenirken hata: {str(e)}"
                }

        # Resmi oku
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        tensor = val_transform(image).unsqueeze(0).to(device)

        # Tahmin
        with torch.no_grad():
            outputs = current_model(tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        class_id = predicted.item()
        class_name = CLASS_NAMES.get(class_id, f"Bilinmiyor ({class_id})")
        score = confidence.item()

        return {
            "className": class_name,
            "confidence": round(score, 4),
            "source_model": current_model_key,
            "message": f"Tespit: {class_name}"
        }
    except Exception as e:
        print(f"HATA: {e}")
        return {"className": "Hata", "confidence": 0.0, "source_model": current_model_key, "message": str(e)}

# --- 8. BAŞLAT ---
ngrok.kill()
ngrok_tunnel = ngrok.connect(8000)
print('\n' + '='*60)
print(f'🚀 LİNKİNİZ: {ngrok_tunnel.public_url}/predict')
print(f'📋 MODEL LİSTESİ: {ngrok_tunnel.public_url}/models')
print('='*60 + '\n')

nest_asyncio.apply()
config = uvicorn.Config(app, port=8000)
server = uvicorn.Server(config)
await server.serve()
