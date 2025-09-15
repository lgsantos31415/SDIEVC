from ultralytics import YOLO
import cv2
import numpy as np

# --- CONFIGURAÇÕES ---
MODEL_ID = "best.pt"         # Caminho para o seu arquivo de modelo treinado (.pt)
WEBCAM_INDEX = 0             # Índice da sua câmera (geralmente 0)
WINDOW_NAME = 'Live PPE Detection (YOLOv8) - Pressione "q" para sair'

# --- CARREGAMENTO DO MODELO ---
try:
    print(f"Carregando o modelo: {MODEL_ID}...")
    model = YOLO(MODEL_ID)
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    print("Verifique se o arquivo 'best.pt' está na mesma pasta que este script.")
    exit()

# --- INICIALIZAÇÃO DA CÂMERA ---
cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print("Erro: Não foi possível abrir a câmera.")
    exit()

# --- PREPARAÇÃO DA JANELA PARA TELA CHEIA ---
cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Tenta obter as dimensões da tela a partir da câmera, com um fallback
try:
    screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if screen_width == 0 or screen_height == 0:
        raise ValueError("Resolução da câmera inválida")
except Exception:
    screen_width = 1920
    screen_height = 1080
    print("Aviso: Não foi possível determinar a resolução da tela. Usando 1920x1080.")

# --- LOOP DE PROCESSAMENTO DE VÍDEO ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro: Fim do stream de vídeo ou erro na captura.")
        break

    # Executa a detecção com YOLOv8
    results = model(frame, stream=True, verbose=False)

    detected_classes_ids = set()

    # Processa e desenha os resultados no frame
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls_id = int(box.cls[0])
            
            detected_classes_ids.add(cls_id)
            
            label = model.names[cls_id]
            text = f"{label}: {conf:.2f}"

            # Desenha a caixa delimitadora e o texto da detecção
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

    # --- DESENHA A LISTA DE STATUS DAS CLASSES COM FUNDO ---
    font_scale = 0.6
    font_thickness = 1
    line_height = int(font_scale * 35)
    margin = 10
    box_width = 200

    # Define a área para a lista (canto superior direito)
    box_x1 = frame.shape[1] - box_width - margin
    box_y1 = margin
    box_h = (len(model.names) * line_height) + margin
    box_y2 = box_y1 + box_h
    box_x2 = frame.shape[1] - margin

    # Cria o fundo semi-transparente
    sub_img = frame[box_y1:box_y2, box_x1:box_x2]
    black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
    alpha = 0.5
    res = cv2.addWeighted(sub_img, 1 - alpha, black_rect, alpha, 1.0)
    frame[box_y1:box_y2, box_x1:box_x2] = res

    # Desenha o nome de cada classe na lista
    label_y_pos = box_y1 + line_height
    for cls_id, cls_name in sorted(model.names.items()):
        color = (0, 255, 0) if cls_id in detected_classes_ids else (128, 128, 128)
        cv2.putText(frame, cls_name, (box_x1 + 10, label_y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
        label_y_pos += line_height

    # --- AJUSTA O FRAME PARA TELA CHEIA MANTENDO A PROPORÇÃO ---
    h, w, _ = frame.shape
    scale = min(screen_width / w, screen_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Centraliza o frame redimensionado em um fundo preto
    fullscreen_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    x_offset = (screen_width - new_w) // 2
    y_offset = (screen_height - new_h) // 2
    fullscreen_frame[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_frame

    # Mostra o frame final
    cv2.imshow(WINDOW_NAME, fullscreen_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- LIMPEZA ---
print("Encerrando...")
cap.release()
cv2.destroyAllWindows()