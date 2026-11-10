import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def imshow(img, title=None, color_img=False, colorbar=True):
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    if title:
        plt.title(title, fontsize=11, fontweight='bold')
    plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()


def obtener_candidatos_por_umbral(imagen_gris, umbral):
    _, img_binaria = cv2.threshold(imagen_gris, umbral, 255, cv2.THRESH_BINARY)
    num_lbl, mapa_lbl, stats, _ = cv2.connectedComponentsWithStats(img_binaria, connectivity=8)

    mascara_area = np.zeros_like(img_binaria)
    mascara_aspecto = np.zeros_like(img_binaria)

    for i in range(1, num_lbl):
        x, y, w, h, area = stats[i]
        if not 15 < area < 200:
            continue
        mascara_area[mapa_lbl == i] = 255
        if 1.5 < h / w < 3.5 and 8 <= h <= 25:
            mascara_aspecto[mapa_lbl == i] = 255

    return img_binaria, mapa_lbl, stats, mascara_area, mascara_aspecto


def agrupar_por_centroides(mascara_aspecto):
    num_lbl, mapa_lbl, stats, centroids = cv2.connectedComponentsWithStats(mascara_aspecto, connectivity=8)

    comps = [
        {'x': int(stats[i][0]), 'y': int(stats[i][1]), 'w': int(stats[i][2]), 'h': int(stats[i][3]),
         'cx': centroids[i][0], 'cy': centroids[i][1], 'label': i}
        for i in range(1, num_lbl)
    ]
    if len(comps) < 5:
        return None, 0, None

    comps.sort(key=lambda c: c['cx'])

    for size in (6, 5):
        if len(comps) < size:
            continue
        best, best_score = None, -1e9
        for i in range(len(comps) - size + 1):
            grupo = comps[i:i + size]
            dy = max(c['cy'] for c in grupo) - min(c['cy'] for c in grupo)
            avg_h = np.mean([c['h'] for c in grupo])
            if dy > 1.2 * avg_h:
                continue
            if any(not -0.4 * c1['w'] <= c2['x'] - (c1['x'] + c1['w']) <= 6.0 * c1['w']
                   for c1, c2 in zip(grupo, grupo[1:])):
                continue
            x0 = min(c['x'] for c in grupo)
            y0 = min(c['y'] for c in grupo)
            x1 = max(c['x'] + c['w'] for c in grupo)
            y1 = max(c['y'] + c['h'] for c in grupo)
            aspect = (x1 - x0) / (y1 - y0)
            if not 2.0 < aspect < 6.0:
                continue
            score = -abs(aspect - 3.5) - dy
            if score > best_score:
                best_score = score
                best = (x0, y0, x1 - x0, y1 - y0, grupo)
        if best:
            x, y, w, h, grupo = best
            mascara = np.zeros_like(mascara_aspecto)
            for c in grupo:
                mascara[mapa_lbl == c['label']] = 255
            return (x, y, w, h), size, mascara

    return None, 0, None


def calcular_umbral_optimo(imagen_gris):
    umbral_5 = None
    for umbral in range(40, 230, 2):
        _, _, _, _, m = obtener_candidatos_por_umbral(imagen_gris, umbral)
        _, n, _ = agrupar_por_centroides(m)
        if n == 6:
            return umbral
        if n == 5 and umbral_5 is None:
            umbral_5 = umbral
    return umbral_5 if umbral_5 is not None else 127


def segmentar_caracteres_placa(imagen_gris, bbox_placa, mascara_placa):
    x, y, w, h = bbox_placa
    _, _, stats_l, _ = cv2.connectedComponentsWithStats(mascara_placa, connectivity=8)
    alto_letra = np.median(stats_l[1:, 3])

    mx, my = int(alto_letra), int(0.4 * alto_letra)
    x0, y0 = max(0, x - mx), max(0, y - my)
    roi = imagen_gris[y0:y + h + my, x0:x + w + mx]

    umbral_b, img_bin_rec, mapa_lbl = 0, None, None
    mejor_parcial = None

    for umbral_b in range(40, 230, 2):
        _, img_bin_rec = cv2.threshold(roi, umbral_b, 255, cv2.THRESH_BINARY)
        num_lbl, mapa_lbl, stats, _ = cv2.connectedComponentsWithStats(img_bin_rec, connectivity=8)

        bboxes = [
            [int(stats[i][0]), int(stats[i][1]), int(stats[i][2]), int(stats[i][3]), i]
            for i in range(1, num_lbl)
            if stats[i][4] > 15
            and 0.7 * alto_letra <= stats[i][3] <= 1.6 * alto_letra
            and 1.2 < stats[i][3] / stats[i][2] < 3.5
        ]
        if len(bboxes) == 6:
            break
        if len(bboxes) == 5 and mejor_parcial is None:
            mejor_parcial = (umbral_b, [c[:] for c in bboxes], img_bin_rec, mapa_lbl)
    else:
        if mejor_parcial is not None:
            umbral_b, bboxes, img_bin_rec, mapa_lbl = mejor_parcial
            bboxes = [c[:] for c in bboxes]

    bboxes.sort(key=lambda c: c[0])

    for k, c in enumerate(bboxes):
        if c[3] > 1.2 * alto_letra and len(bboxes) > 1:
            vecino = bboxes[k + 1] if k + 1 < len(bboxes) else bboxes[k - 1]
            arriba, abajo = vecino[1] - 1, vecino[1] + vecino[3] + 1
            mask_c = (mapa_lbl == c[4]).astype(np.uint8)
            mask_c[:arriba] = 0
            mask_c[abajo:] = 0
            ys, xs = np.nonzero(mask_c)
            if len(xs) > 0 and len(ys) > 0:
                c[:4] = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1),
                         int(ys.max() - ys.min() + 1)]

    mascara = np.zeros_like(roi)
    for cx, cy, cw, ch, i in bboxes:
        caja = mapa_lbl[cy:cy + ch, cx:cx + cw] == i
        mascara[cy:cy + ch, cx:cx + cw][caja] = 255

    return roi, img_bin_rec, umbral_b, bboxes, (x0, y0), alto_letra, mascara


def procesar_patente(nombre, directorio):
    img_bgr = cv2.imread(os.path.join(directorio, nombre))
    if img_bgr is None:
        return None

    img_gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    umbral_a = calcular_umbral_optimo(img_gris)
    img_bin_a, _, stats_a, mascara_area, mascara_aspecto = obtener_candidatos_por_umbral(img_gris, umbral_a)

    bbox_placa, num_letras, mascara_placa = agrupar_por_centroides(mascara_aspecto)

    img_componentes = cv2.cvtColor(img_gris, cv2.COLOR_GRAY2BGR)
    for i in range(1, len(stats_a)):
        x, y, w, h, area = stats_a[i]
        if area > 5:
            cv2.rectangle(img_componentes, (x, y), (x + w, y + h), (0, 255, 0), 1)

    vis_grupo = cv2.cvtColor(mascara_aspecto, cv2.COLOR_GRAY2BGR)
    if bbox_placa:
        x_p, y_p, w_p, h_p = bbox_placa
        cv2.rectangle(vis_grupo, (x_p, y_p), (x_p + w_p, y_p + h_p), (0, 255, 0), 2)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1), imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), "1. Original", True, False)
    plt.subplot(1, 2, 2), imshow(img_gris, "2. Grises", False, False)
    plt.tight_layout(); plt.show(block=True)

    plt.figure(figsize=(8, 6))
    imshow(img_bin_a, f"3. Binarizada (U={umbral_a})", False, False)
    plt.tight_layout(); plt.show(block=True)

    plt.figure(figsize=(8, 6))
    imshow(img_componentes, f"4. Componentes ({len(stats_a) - 1})", True, False)
    plt.tight_layout(); plt.show(block=True)

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1), imshow(img_bin_a, "5. Sin filtros", False, False)
    plt.subplot(1, 3, 2), imshow(mascara_area, "6. Filtro área", False, False)
    plt.subplot(1, 3, 3), imshow(mascara_aspecto, "7. Filtro aspecto", False, False)
    plt.tight_layout(); plt.show(block=True)

    if bbox_placa is None:
        print(f"{nombre}: no se detectó placa.")
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "NO SE DETECTÓ PLACA", ha='center', va='center',
                 fontsize=20, color='red', fontweight='bold')
        plt.axis('off'); plt.show(block=True)
        return None

    roi, img_bin_rec, umbral_b, bboxes, (x0, y0), alto_letra, mascara_car = \
        segmentar_caracteres_placa(img_gris, bbox_placa, mascara_placa)

    vis_car = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    for cx, cy, cw, ch, _ in bboxes:
        cv2.rectangle(vis_car, (cx, cy), (cx + cw - 1, cy + ch - 1), (0, 255, 0), 1)

    px0 = x0 + min(c[0] for c in bboxes)
    py0 = y0 + min(c[1] for c in bboxes)
    px1 = x0 + max(c[0] + c[2] for c in bboxes)
    py1 = y0 + max(c[1] + c[3] for c in bboxes)
    margen = int(0.3 * alto_letra)

    img_final = img_bgr.copy()
    cv2.rectangle(img_final, (px0 - margen, py0 - margen),
                  (px1 + margen, py1 + margen), (0, 255, 0), 2)
    for cx, cy, cw, ch, _ in bboxes:
        cv2.rectangle(img_final, (x0 + cx, y0 + cy),
                      (x0 + cx + cw - 1, y0 + cy + ch - 1), (0, 255, 0), 1)

    print(f"{nombre}: Umbral A={umbral_a} ({num_letras} letras) | "
          f"Umbral B={umbral_b} | Caracteres={len(bboxes)}")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1), imshow(mascara_aspecto, "8. Componentes", False, False)
    plt.subplot(1, 2, 2), imshow(vis_grupo, f"9. Agrupación ({num_letras})", True, False)
    plt.tight_layout(); plt.show(block=True)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1), imshow(roi, "10. Recorte", False, False)
    plt.subplot(1, 2, 2), imshow(img_bin_rec, f"11. Umbralizado ({umbral_b})", False, False)
    plt.tight_layout(); plt.show(block=True)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1), imshow(vis_car, "12. Caracteres", True, False)
    plt.subplot(1, 2, 2), imshow(cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB),
                                 "13. Resultado final", True, False)
    plt.tight_layout(); plt.show(block=True)

    escala = 4
    ancho_p, alto_p = px1 - px0, py1 - py0
    rx0, ry0 = max(0, px0 - int(0.3 * ancho_p)), max(0, py0 - int(1.0 * alto_p))
    rx1, ry1 = px1 + int(0.3 * ancho_p), py1 + int(1.0 * alto_p)

    img_resumen = cv2.resize(img_bgr[ry0:ry1, rx0:rx1], None,
                             fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
    for cx, cy, cw, ch, _ in bboxes:
        ax0, ay0 = (x0 + cx - rx0) * escala, (y0 + cy - ry0) * escala
        cv2.rectangle(img_resumen, (ax0, ay0),
                      (ax0 + cw * escala, ay0 + ch * escala), (0, 255, 0), 2)

    return nombre, img_resumen, len(bboxes), umbral_b, roi, bboxes


if __name__ == '__main__':
    directorio = './Imagenes/Iniciales/Patentes/'
    archivos = sorted(f for f in os.listdir(directorio) if f.endswith('.png'))
    resultados = [r for r in (procesar_patente(a, directorio) for a in archivos) if r]

    if resultados:
        total_chars = sum(r[2] for r in resultados)
        total_posibles = 72
        porcentaje = total_chars / total_posibles * 100

        n = len(resultados)
        max_chars = max(max((r[2] for r in resultados), default=1), 1)

        fig = plt.figure(figsize=(14, 3.0 * n))
        outer = fig.add_gridspec(n, 1, hspace=0.45)

        for i, (nombre, img_resumen, n_car, umbral_b, roi, bboxes) in enumerate(resultados):
            inner = outer[i].subgridspec(
                2, max_chars,
                height_ratios=[3, 1.5],
                hspace=0.15, wspace=0.05,
            )

            ax_plate = fig.add_subplot(inner[0, :])
            ax_plate.imshow(cv2.cvtColor(img_resumen, cv2.COLOR_BGR2RGB))
            ax_plate.set_title(f"{nombre} — {n_car} car. (T={umbral_b})",
                               fontsize=11, fontweight='bold')
            ax_plate.axis('off')

            for j, (cx, cy, cw, ch, _) in enumerate(bboxes):
                ax_c = fig.add_subplot(inner[1, j])
                ax_c.imshow(roi[cy:cy + ch, cx:cx + cw], cmap='gray',
                            vmin=0, vmax=255, interpolation='nearest')
                ax_c.axis('off')

            for j in range(len(bboxes), max_chars):
                fig.add_subplot(inner[1, j]).axis('off')

        fig.suptitle(
            f"Segmentación de Patentes — {total_chars}/{total_posibles} "
            f"caracteres correctos ({porcentaje:.1f}%)",
            fontsize=15, fontweight='bold',
        )
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show(block=True)