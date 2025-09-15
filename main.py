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

# --- MAPEAMENTO DE CLASSES (FEITO UMA VEZ) ---
# Dicionário para armazenar os IDs das classes que nos interessam
# O valor será -1 se a classe não for encontrada no modelo
class_ids = {
    "Hardhat": -1, "Mask": -1, "NO-Hardhat": -1, "NO-Mask": -1,
    "NO-Safety Vest": -1, "Person": -1, "Safety Vest": -1
}
for k, v in model.names.items():
    if v in class_ids:
        class_ids[v] = k
for name, class_id in class_ids.items():
    if class_id == -1:
        print(f"Aviso: A classe '{name}' não foi encontrada no modelo. A lógica associada pode falhar.")

# --- INICIALIZAÇÃO DA CÂMERA ---
cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print("Erro: Não foi possível abrir a câmera.")
    exit()

# --- PREPARAÇÃO DA JANELA PARA TELA CHEIA ---
cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

try:
    screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if screen_width == 0 or screen_height == 0: raise ValueError("Resolução inválida")
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

    results = model(frame, stream=True, verbose=False)

    detected_classes_ids = set()
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            detected_classes_ids.add(cls_id)
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            label = model.names[cls_id]
            text = f"{label}: {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

    # --- LÓGICA DE VERIFICAÇÃO DE EPI ---
    person_detected = class_ids["Person"] in detected_classes_ids
    hardhat_ok = class_ids["Hardhat"] in detected_classes_ids
    mask_ok = class_ids["Mask"] in detected_classes_ids
    vest_ok = class_ids["Safety Vest"] in detected_classes_ids
    
    no_hardhat = class_ids["NO-Hardhat"] in detected_classes_ids
    no_mask = class_ids["NO-Mask"] in detected_classes_ids
    no_vest = class_ids["NO-Safety Vest"] in detected_classes_ids

    is_correct = False
    if person_detected:
        wears_all_ppe = hardhat_ok and mask_ok and vest_ok
        any_no_ppe = no_hardhat or no_mask or no_vest
        is_correct = wears_all_ppe and not any_no_ppe

    # --- DESENHA A LISTA DE STATUS DAS CLASSES COM FUNDO ---
    font_scale = 0.5
    font_thickness = 1
    line_height = int(font_scale * 45)
    margin = 10
    box_width = 200

    ppe_list = [
        ("Capacete", "Hardhat", "NO-Hardhat"), ("Mascara", "Mask", "NO-Mask"),
        ("Colete", "Safety Vest", "NO-Safety Vest"), ("Pessoa", "Person", None)
    ]
    num_lines = len(ppe_list) + 1 if person_detected else len(ppe_list)
    
    box_x1 = frame.shape[1] - box_width - margin
    box_y1 = margin
    box_h = (num_lines * line_height)
    box_y2 = min(box_y1 + box_h, frame.shape[0] - 1)
    box_x2 = frame.shape[1] - margin

    sub_img = frame[box_y1:box_y2, box_x1:box_x2]
    if sub_img.size > 0:
        black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
        res = cv2.addWeighted(sub_img, 0.5, black_rect, 0.5, 1.0)
        frame[box_y1:box_y2, box_x1:box_x2] = res

    label_y_pos = box_y1 + line_height - (margin // 2)

    for display_name, pos_class, neg_class in ppe_list:
        is_detected = class_ids.get(pos_class, -1) in detected_classes_ids
        is_neg_detected = neg_class and class_ids.get(neg_class, -1) in detected_classes_ids
        
        text_color = (0, 255, 0) if is_detected else (200, 200, 200)

        if is_neg_detected:
            text_color = (255, 255, 255)
            line_y1 = label_y_pos - line_height + margin
            line_y2 = label_y_pos + (margin//2)
            line_roi = frame[line_y1:line_y2, box_x1:box_x2]
            if line_roi.size > 0:
                red_rect = np.full(line_roi.shape, (0, 0, 200), dtype=np.uint8)
                line_res = cv2.addWeighted(line_roi, 0.4, red_rect, 0.6, 1.0)
                frame[line_y1:line_y2, box_x1:box_x2] = line_res
        
        cv2.putText(frame, display_name, (box_x1 + 10, label_y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
        label_y_pos += line_height

    if person_detected:
        separator_y = label_y_pos - line_height
        cv2.line(frame, (box_x1, separator_y), (box_x2, separator_y), (255, 255, 255), 1)

        status_text = "Status: Correto" if is_correct else "Status: Incorreto"
        status_color = (0, 255, 0) if is_correct else (255, 255, 255)

        if not is_correct:
            line_y1 = label_y_pos - line_height + (margin//2)
            line_y2 = label_y_pos + margin
            line_roi = frame[line_y1:line_y2, box_x1:box_x2]
            if line_roi.size > 0:
                red_rect = np.full(line_roi.shape, (0, 0, 200), dtype=np.uint8)
                line_res = cv2.addWeighted(line_roi, 0.2, red_rect, 0.8, 1.0)
                frame[line_y1:line_y2, box_x1:box_x2] = line_res
        
        cv2.putText(frame, status_text, (box_x1 + 10, label_y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, status_color, font_thickness)

    # --- AJUSTA O FRAME PARA TELA CHEIA MANTENDO A PROPORÇÃO ---
    h, w, _ = frame.shape
    scale = min(screen_width / w, screen_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    fullscreen_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    x_offset = (screen_width - new_w) // 2
    y_offset = (screen_height - new_h) // 2
    fullscreen_frame[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_frame

    cv2.imshow(WINDOW_NAME, fullscreen_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- LIMPEZA ---
print("Encerrando...")
cap.release()
cv2.destroyAllWindows()