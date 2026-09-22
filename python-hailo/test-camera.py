import time

import cv2
from PIL import Image
from picamera2 import Picamera2
from trustmark_hailo import TrustMark


MODEL_TYPE = "Q"
DECODE_MODE = "binary"

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1280, 720), "format": "BGR888"})
picam2.configure(config)
picam2.start()

tm = TrustMark(model_type=MODEL_TYPE, loadBBoxDetector=True, verbose=True)
print("Press 'q' in the preview window to quit.")

last_t = time.time()
try:
    while True:
        # picamera2 quirk, format BGR888 actually returns RGB channel order
        frame_rgb = picam2.capture_array()
        pil_frame = Image.fromarray(frame_rgb, mode="RGB")
        w, h = pil_frame.size

        boxes = tm.localize(pil_frame, return_all=True) or []
        display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)  # cv2.imshow wants BGR

        for x1, y1, x2, y2 in boxes:
            px1, py1 = max(0, int(x1 * w)), max(0, int(y1 * h))
            px2, py2 = min(w, int(x2 * w)), min(h, int(y2 * h))
            if px2 <= px1 or py2 <= py1:
                continue

            crop = pil_frame.crop((px1, py1, px2, py2))
            secret, detected, schema = tm.subimage_decode(crop, MODE=DECODE_MODE)
            color = (0, 255, 0) if detected else (0, 165, 255)
            cv2.rectangle(display, (px1, py1), (px2, py2), color, 2)

            if detected:
                print(f"[{time.strftime('%H:%M:%S')}] DETECTED: secret={secret!r} schema={schema} box=({px1},{py1})-({px2},{py2})")
                cv2.putText(display, secret, (px1, max(py1 - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        now = time.time()
        fps = 1.0 / max(now - last_t, 1e-6)
        last_t = now
        cv2.putText(display, f"{fps:.2f} fps  {len(boxes)} box(es)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow(f"TrustMark live detect ({MODEL_TYPE})", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    tm.close()
    picam2.stop()
    cv2.destroyAllWindows()
