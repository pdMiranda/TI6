import cv2
import numpy as np
import imutils
from pathlib import Path

def crop_img(img: np.ndarray) -> np.ndarray:
    """
    Função de pré-processamento para cortar a imagem e focar na região do cérebro.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    if not cnts:
        return img

    c = max(cnts, key=cv2.contourArea)

    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    new_img = img[extTop[1]:extBot[1], extLeft[0]:extRight[0]].copy()
    return new_img

def process_and_save_dataset(source_dir: Path, dest_dir: Path, img_size: int):
    """
    Aplica o pré-processamento (corte e redimensionamento) nas imagens de um diretório
    e salva o resultado em um novo diretório.
    """
    print(f"Iniciando pré-processamento de '{source_dir}' para '{dest_dir}'...")
    labels = [d.name for d in source_dir.iterdir() if d.is_dir()]
    
    for label in labels:
        source_label_dir = source_dir / label
        dest_label_dir = dest_dir / label
        dest_label_dir.mkdir(parents=True, exist_ok=True)

        for img_file in source_label_dir.glob('*'):
            try:
                image = cv2.imread(str(img_file))
                if image is None:
                    print(f"Aviso: Não foi possível ler a imagem {img_file}")
                    continue
                
                cropped_image = crop_img(image)
                resized_image = cv2.resize(cropped_image, (img_size, img_size))
                
                save_path = dest_label_dir / img_file.name
                cv2.imwrite(str(save_path), resized_image)
            except Exception as e:
                print(f"Erro ao processar {img_file}: {e}")
                
    print(f"Pré-processamento e salvamento concluídos em '{dest_dir}'.")