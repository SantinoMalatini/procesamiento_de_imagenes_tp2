import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- 1. Funciones Auxiliares --------------------------------------------------

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    
    if title is not None:
        plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)

def calcular_inercia(data, k):
    if k == 1:
        centro = np.mean(data)
        return np.sum((data - centro)**2)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centros = cv2.kmeans(
        data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    inercia = 0
    for i in range(k):
        cluster_data = data[labels.flatten() == i]
        inercia += np.sum((cluster_data - centros[i])**2)
    return inercia

def calcular_k_optimo_gap(data, max_k=4, b=15):
    n = len(data)
    if n <= 1: return 1
    max_k = min(max_k, n - 1)
    if max_k < 1: return 1

    gaps = np.zeros(max_k)
    s_k = np.zeros(max_k)
    inercia_obs = np.zeros(max_k)

    for k in range(1, max_k + 1):
        inercia_obs[k-1] = calcular_inercia(data, k)

    min_val, max_val = data.min(), data.max()
    inercia_refs = np.zeros((b, max_k))

    for i in range(b):
        ref_data = np.random.uniform(min_val, max_val, size=data.shape).astype(np.float32)
        for k in range(1, max_k + 1):
            inercia_refs[i, k-1] = calcular_inercia(ref_data, k)

    for k in range(max_k):
        obs_val = inercia_obs[k] if inercia_obs[k] > 0 else 1e-10
        ref_logs = np.log(inercia_refs[:, k] + 1e-10)
        gaps[k] = np.mean(ref_logs) - np.log(obs_val)
        sd = np.std(ref_logs)
        s_k[k] = sd * np.sqrt(1 + 1.0 / b)

    for k in range(max_k - 1):
        if gaps[k] >= gaps[k+1] - s_k[k+1]:
            return k + 1
    return max_k


# --- 2. Algoritmo Principal ---------------------------------------------------

def extraer_puntos_dado(img_gray, bbox, hull):
    x, y, w, h = bbox
    margen = 5

    y0, y1 = max(0, y - margen), min(img_gray.shape[0], y + h + margen)
    x0, x1 = max(0, x - margen), min(img_gray.shape[1], x + w + margen)
    roi_gray = img_gray[y0:y1, x0:x1]

    mask_dado = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    c_local = hull.copy()
    c_local[:, 0, 0] -= x0
    c_local[:, 0, 1] -= y0
    cv2.drawContours(mask_dado, [c_local], -1, 255, -1)

    kernel_erode = np.ones((5, 5), np.uint8)
    mask_interior = cv2.erode(mask_dado, kernel_erode, iterations=4)

    roi_limpio = np.ones_like(roi_gray) * 255
    roi_limpio[mask_interior == 255] = roi_gray[mask_interior == 255]

    blur = cv2.GaussianBlur(roi_gray, (5, 5), 1.0)
    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.bitwise_and(edges, mask_interior)
    
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(edges, kernel_morph, iterations=1)
    
    im_floodfill = dilated.copy()
    h_roi, w_roi = dilated.shape[:2]
    mask_ff = np.zeros((h_roi + 2, w_roi + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask_ff, (0, 0), 255)
    filled = dilated | cv2.bitwise_not(im_floodfill)
    
    eroded = cv2.erode(filled, kernel_morph, iterations=1)

    cnts_pips, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    puntos_validos = 0
    pips_contours = []
    
    for c in cnts_pips:
        area = cv2.contourArea(c)
        if 500 < area < 5000:
            perim = cv2.arcLength(c, True)
            if area > 0:
                ff = (perim ** 2) / area
                if 10.0 <= ff <= 18.0: 
                    puntos_validos += 1
                    pips_contours.append(c)

    datos_visuales = {
        'roi': roi_gray,
        'mask_dado': mask_dado,
        'roi_limpio': roi_limpio,
        'thresh': edges,
        'thresh_clean': eroded,
        'pips': pips_contours,
        'count': puntos_validos,
    }
    return puntos_validos, datos_visuales


def procesar_monedas_y_dados(ruta_imagen):
    img_color = cv2.imread(ruta_imagen)
    if img_color is None:
        print(f"Error: No se pudo cargar la imagen en la ruta: {ruta_imagen}")
        return None
    
    img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    img_blur = cv2.GaussianBlur(img_gray, (9, 9), 2.0)
    edges = cv2.Canny(img_blur, 20, 80)
    
    kernel_borde = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    img_dilated = cv2.dilate(edges, kernel_borde, iterations=3)
    
    im_floodfill = img_dilated.copy()
    h_img, w_img = img_dilated.shape[:2]
    mask_ff = np.zeros((h_img + 2, w_img + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask_ff, (0, 0), 255)
    img_filled = img_dilated | cv2.bitwise_not(im_floodfill)
    
    img_eroded = cv2.erode(img_filled, kernel_borde, iterations=3)
    
    kernel_apertura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    img_mask = cv2.morphologyEx(img_eroded, cv2.MORPH_OPEN, kernel_apertura)

    contornos, _ = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    monedas = []
    dados = []
    
    for c in contornos:
        area = cv2.contourArea(c)
        if area > 1500:
            hull = cv2.convexHull(c)
            x, y, w, h = cv2.boundingRect(hull)
            perim = cv2.arcLength(hull, True)
            area_hull = cv2.contourArea(hull)
            
            ff = (perim ** 2) / area_hull if area_hull > 0 else 0
            (cx, cy), radius = cv2.minEnclosingCircle(hull)
            
            obj_data = {'bbox': (x, y, w, h), 'radius': radius, 'hull': hull, 'area': area_hull}
            
            if ff < 13.5:
                monedas.append(obj_data)
            else:
                dados.append(obj_data)

    k_optimo = 1
    if monedas:
        areas_monedas = np.array([[m['area']] for m in monedas], dtype=np.float32)
        k_optimo = calcular_k_optimo_gap(areas_monedas, max_k=4)

        if k_optimo > 1:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, labels, centros = cv2.kmeans(
                areas_monedas, k_optimo, None, criteria, 10, cv2.KMEANS_PP_CENTERS
            )
            centros = centros.flatten()
            orden_clusters = np.argsort(centros)
            
            nombres = ["Tamaño 1 (Pequeña)", "Tamaño 2 (Media)", "Tamaño 3 (Grande)", "Tamaño 4 (Extra)"]
            mapa_tamanos = {int(c): nombres[i] for i, c in enumerate(orden_clusters)}

            for i, m in enumerate(monedas):
                m['tipo_moneda'] = mapa_tamanos[int(labels[i][0])]
        else:
            for m in monedas:
                m['tipo_moneda'] = "Único tamaño"

    datos_visuales_dados = []
    for idx, d in enumerate(dados, 1):
        puntos, datos_vis = extraer_puntos_dado(img_gray, d['bbox'], d['hull'])
        d['valor_dado'] = puntos
        datos_vis['idx'] = idx
        datos_visuales_dados.append(datos_vis)

    return img_rgb, img_gray, img_blur, edges, img_dilated, img_filled, img_eroded, img_mask, monedas, dados, k_optimo, datos_visuales_dados


# --- 3. Ejecución y Exploración de Resultados ---------------------------------

if __name__ == '__main__':
    ruta_archivo = './Imagenes/Iniciales/monedas.jpg'
    
    resultados = procesar_monedas_y_dados(ruta_archivo)
    if resultados is not None:
        img_rgb, img_gray, img_blur, edges, img_dilated, img_filled, img_eroded, img_mask, monedas, dados, k_optimo, datos_visuales_dados = resultados

        # 3.1 Preprocesamiento (4 etapas juntas, bloqueante)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(img_rgb); axes[0].set_title("1. Original"); axes[0].axis('off')
        axes[1].imshow(img_gray, cmap='gray'); axes[1].set_title("2. Grises"); axes[1].axis('off')
        axes[2].imshow(img_blur, cmap='gray'); axes[2].set_title("3. Pasabajos (Blur)"); axes[2].axis('off')
        axes[3].imshow(edges, cmap='gray'); axes[3].set_title("4. Bordes (Canny)"); axes[3].axis('off')
        plt.tight_layout()
        plt.show(block=True)

        # 3.2 Morfología y Máscaras (4 etapas juntas, bloqueante)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(img_dilated, cmap='gray'); axes[0].set_title("5. Dilatación"); axes[0].axis('off')
        axes[1].imshow(img_filled, cmap='gray'); axes[1].set_title("6. Relleno de Huecos"); axes[1].axis('off')
        axes[2].imshow(img_eroded, cmap='gray'); axes[2].set_title("7. Erosión"); axes[2].axis('off')
        axes[3].imshow(img_mask, cmap='gray'); axes[3].set_title("8. Apertura (Máscara Final)"); axes[3].axis('off')
        plt.tight_layout()
        plt.show(block=True)

        # 3.3 Clasificación General (Monedas vs Dados, bloqueante)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img_rgb)
        ax.set_title("9. Clasificación: Azul=Monedas, Verde=Dados")
        ax.axis('off')
        for m in monedas:
            x, y, w, h = m['bbox']
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='blue', lw=2))
        for d in dados:
            x, y, w, h = d['bbox']
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='lime', lw=2))
        plt.tight_layout()
        plt.show(block=True)

        # 3.4 Monedas por Tamaño (K-Means + GAP, bloqueante)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img_rgb)
        ax.set_title(f"10. Clusters de Monedas (K={k_optimo} calculado con GAP)")
        ax.axis('off')
        colores_monedas = ['cyan', 'yellow', 'magenta', 'red']
        
        for m in monedas:
            x, y, w, h = m['bbox']
            label_text = m['tipo_moneda']
            color_idx = int(label_text.split(" ")[1]) - 1 if "Tamaño" in label_text else 0
            color_moneda = colores_monedas[color_idx % len(colores_monedas)]
            
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color_moneda, lw=2))
            ax.text(x, y - 5, label_text, color='black', fontsize=9, fontweight='bold',
                    bbox=dict(facecolor=color_moneda, alpha=0.8, edgecolor='none', pad=2))
        plt.tight_layout()
        plt.show(block=True)

        # 3.5 Análisis Detallado de Dados (de a uno por cada dado, bloqueantes)
        if datos_visuales_dados:
            for datos in datos_visuales_dados:
                fig_d, axes_d = plt.subplots(1, 5, figsize=(15, 3))
                fig_d.suptitle(f"11. Dado #{datos['idx']} - Puntos detectados: {datos['count']}", fontsize=12, fontweight='bold')
                
                axes_d[0].imshow(datos['roi'], cmap='gray')
                axes_d[0].set_title("ROI"); axes_d[0].axis('off')

                axes_d[1].imshow(datos['roi_limpio'], cmap='gray')
                axes_d[1].set_title("ROI limpio"); axes_d[1].axis('off')
                
                axes_d[2].imshow(datos['thresh'], cmap='gray')
                axes_d[2].set_title("Bordes (Canny)"); axes_d[2].axis('off')

                axes_d[3].imshow(datos['thresh_clean'], cmap='gray')
                axes_d[3].set_title("Relleno Puntos"); axes_d[3].axis('off')

                roi_color = cv2.cvtColor(datos['roi'], cv2.COLOR_GRAY2RGB)
                cv2.drawContours(roi_color, datos['pips'], -1, (0, 255, 0), -1)
                axes_d[4].imshow(roi_color)
                axes_d[4].set_title(f"Total: {datos['count']}", color='green', fontweight='bold')
                axes_d[4].axis('off')

                plt.tight_layout()
                plt.show(block=True)