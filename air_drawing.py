"""
AIR DRAWING SYSTEM - COMPLETE EDITION
=======================================
GESTURES:
- ☝️  Index only         → Draw / shape start point
- ✊  Fist               → Commit shape / drop object
- ✌️  Index + Middle     → Hover mode (colors, +/- buttons, select tool)

KEYBOARD:
- 1=FreeDraw  2=Line  3=Circle  4=Rectangle  5=Triangle  6=Text  7=Select&Move
- B=Background  Z=Undo  C=Clear  S=Save  Q=Quit
"""

import cv2, mediapipe as mp, numpy as np
import os, time, math, urllib.request
from datetime import datetime
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── MODEL ──────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH)
    print("✅ Done!")

# ── CONFIG ──────────────────────────────────────
CANVAS_ALPHA  = 0.85
MIN_BRUSH     = 2
MAX_BRUSH     = 80
ERASER_SIZE   = 50
SAVE_DIR      = "saved_drawings"
HEADER_HEIGHT = 160   # toolbar height
HOVER_TIME    = 0.8   # seconds to hover to activate
MAX_UNDO      = 20

COLORS = [
    ("Red",    (0,   0,   255)),
    ("Orange", (0,  165,  255)),
    ("Yellow", (0,  255,  255)),
    ("Green",  (0,  200,    0)),
    ("Blue",   (255,  50,   0)),
    ("Purple", (180,   0,  180)),
    ("White",  (255, 255,  255)),
    ("Eraser", (0,    0,    0)),
]

TOOLS = {
    '1':'FreeDraw','2':'Line','3':'Circle',
    '4':'Rectangle','5':'Triangle','6':'Text','7':'Select'
}

HAND_CONN = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
]

# ── HAND DETECTOR ───────────────────────────────
class HandDetector:
    def __init__(self):
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.6)
        self.lm  = mp_vision.HandLandmarker.create_from_options(opts)
        self.det = None

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.det = self.lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if self.det.hand_landmarks:
            h,w = frame.shape[:2]
            for hand in self.det.hand_landmarks:
                pts = [(int(lm.x*w),int(lm.y*h)) for lm in hand]
                for a,b in HAND_CONN: cv2.line(frame,pts[a],pts[b],(0,180,0),2)
                for pt in pts: cv2.circle(frame,pt,3,(0,255,0),-1)
        return frame

    def landmarks(self, frame):
        h,w = frame.shape[:2]
        if self.det and self.det.hand_landmarks:
            return [(int(lm.x*w),int(lm.y*h)) for lm in self.det.hand_landmarks[0]]
        return []

    def fingers_up(self, pos):
        if len(pos)<21: return [False]*5
        tips=[4,8,12,16,20]; pip=[3,6,10,14,18]
        up=[pos[tips[0]][0]<pos[pip[0]][0]]
        for i in range(1,5): up.append(pos[tips[i]][1]<pos[pip[i]][1])
        return up

    def is_fist(self, f): return not any(f[1:5])

# ── TOOLBAR ─────────────────────────────────────
class Toolbar:
    """
    Layout (left→right):
      [Color swatches ... ] [ + ] [ - ] [brush circle]
    Bottom row: tool buttons
    """
    def __init__(self, w):
        self.w = w
        self.h = HEADER_HEIGHT
        self.n = len(COLORS)
        # Reserve right 120px for +/- brush buttons
        self.swatch_w   = (w - 160) // self.n
        self.btn_plus   = (w-155, 8,  w-85, self.h-55)   # x1,y1,x2,y2
        self.btn_minus  = (w- 75, 8,  w- 5, self.h-55)
        # Hover state
        self.hover_item = None   # ('color',idx) | ('plus') | ('minus') | ('tool',key)
        self.hover_t    = 0.0

    def draw(self, frame, active_idx, tool, brush, bg_white,
             hover_item=None, hover_prog=0.0):
        cv2.rectangle(frame,(0,0),(self.w,self.h),(28,28,28),-1)

        # ── Color swatches ──
        for i,(label,bgr) in enumerate(COLORS):
            x1 = i*self.swatch_w
            x2 = x1+self.swatch_w
            cv2.rectangle(frame,(x1+4,6),(x2-4,self.h-55),bgr,-1)
            if i==active_idx:
                cv2.rectangle(frame,(x1+2,2),(x2-2,self.h-53),(255,255,255),3)
            if hover_item==('color',i) and hover_prog>0:
                cx,cy=(x1+x2)//2,(self.h-55)//2+6
                cv2.ellipse(frame,(cx,cy),(18,18),-90,0,int(360*hover_prog),(0,255,255),3)
            cv2.putText(frame,label,(x1+4,self.h-38),
                        cv2.FONT_HERSHEY_SIMPLEX,0.33,(210,210,210),1)

        # ── + button ──
        bx1,by1,bx2,by2 = self.btn_plus
        plus_active = hover_item==('plus')
        col = (0,200,80) if plus_active else (50,50,50)
        cv2.rectangle(frame,(bx1,by1),(bx2,by2),col,-1)
        cv2.rectangle(frame,(bx1,by1),(bx2,by2),(150,150,150),1)
        cv2.putText(frame,"+",(bx1+18,by2-14),cv2.FONT_HERSHEY_SIMPLEX,1.4,(255,255,255),2)
        if plus_active and hover_prog>0:
            cx,cy=(bx1+bx2)//2,(by1+by2)//2
            cv2.ellipse(frame,(cx,cy),(32,32),-90,0,int(360*hover_prog),(0,255,80),3)

        # ── − button ──
        bx1,by1,bx2,by2 = self.btn_minus
        minus_active = hover_item==('minus')
        col = (0,80,200) if minus_active else (50,50,50)
        cv2.rectangle(frame,(bx1,by1),(bx2,by2),col,-1)
        cv2.rectangle(frame,(bx1,by1),(bx2,by2),(150,150,150),1)
        cv2.putText(frame,"-",(bx1+22,by2-10),cv2.FONT_HERSHEY_SIMPLEX,1.6,(255,255,255),2)
        if minus_active and hover_prog>0:
            cx,cy=(bx1+bx2)//2,(by1+by2)//2
            cv2.ellipse(frame,(cx,cy),(32,32),-90,0,int(360*hover_prog),(0,80,255),3)

        # ── Brush preview circle ──
        cv2.circle(frame,(self.w-160+10+35,(self.h-55)//2+6),
                   max(1,brush//2),COLORS[active_idx][1],-1)
        cv2.putText(frame,f"{brush}px",(self.w-160+10,self.h-38),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,(180,180,180),1)

        # ── Tool buttons (bottom row) ──
        tool_list = ["1:Draw","2:Line","3:Circle","4:Rect","5:Tri","6:Text","7:Select"]
        tw = self.w//len(tool_list)
        for i,t in enumerate(tool_list):
            x1=i*tw; act=(TOOLS.get(t[0])==tool)
            bg_col=(0,140,255) if act else (45,45,45)
            cv2.rectangle(frame,(x1+2,self.h-48),(x1+tw-2,self.h-2),bg_col,-1 if act else 1)
            if hover_item==('tool',t[0]) and hover_prog>0:
                cx=(x1+x1+tw)//2; cy=self.h-25
                cv2.ellipse(frame,(cx,cy),(tw//2-4,18),-90,0,int(360*hover_prog),(255,200,0),2)
            cv2.putText(frame,t,(x1+4,self.h-14),cv2.FONT_HERSHEY_SIMPLEX,0.38,
                        (255,255,255) if act else (110,110,110),1)

        bg_str="WHITE BG" if bg_white else "CAMERA BG"
        cv2.putText(frame,f"{bg_str} | B=BG  Z=Undo  C=Clear  S=Save  Q=Quit",
                    (8,self.h+22),cv2.FONT_HERSHEY_SIMPLEX,0.43,(140,140,140),1)

    def hit_test(self, x, y):
        """Return what UI element (x,y) is over, or None."""
        # Color swatches
        if 6<=y<=self.h-55:
            if x < self.n*self.swatch_w:
                return ('color', x//self.swatch_w)
            bx1,by1,bx2,by2=self.btn_plus
            if bx1<=x<=bx2 and by1<=y<=by2: return ('plus',)
            bx1,by1,bx2,by2=self.btn_minus
            if bx1<=x<=bx2 and by1<=y<=by2: return ('minus',)
        # Tool buttons
        if self.h-48<=y<=self.h-2:
            tool_list=["1:Draw","2:Line","3:Circle","4:Rect","5:Tri","6:Text","7:Select"]
            tw=self.w//len(tool_list)
            idx=x//tw
            if idx<len(tool_list):
                return ('tool', tool_list[idx][0])
        return None

# ── CANVAS ──────────────────────────────────────
class Canvas:
    def __init__(self, h, w):
        self.h=h; self.w=w
        self.img=np.zeros((h,w,3),dtype=np.uint8)
        self.history=[]; self._new=True
        # Selection / move state
        self.sel_patch  = None   # grabbed image patch
        self.sel_x      = 0
        self.sel_y      = 0

    def snapshot(self):
        if self._new:
            self.history.append(self.img.copy())
            if len(self.history)>MAX_UNDO: self.history.pop(0)
            self._new=False

    def end_stroke(self): self._new=True

    # ── Primitives ──
    def draw_line(self,p1,p2,col,thick):
        if p1 and p2: cv2.line(self.img,p1,p2,col,thick)

    def draw_circle(self,p1,p2,col,thick):
        if p1 and p2:
            r=int(math.hypot(p2[0]-p1[0],p2[1]-p1[1]))
            if r>0: cv2.circle(self.img,p1,r,col,thick)

    def draw_rect(self,p1,p2,col,thick):
        if p1 and p2: cv2.rectangle(self.img,p1,p2,col,thick)

    def _tri_pts(self, p1, p2):
        # p1 = top vertex (first tap), p2 = bottom-right corner
        # p3 = bottom-left: mirror of p2 around p1.x → always symmetric, never cut
        ax, ay = p1
        bx, by = p2
        cx = int(2 * ax - bx)
        cy = by
        return np.array([[ax, ay], [bx, by], [cx, cy]], np.int32)

    def draw_triangle(self, p1, p2, col, thick):
        if not (p1 and p2): return
        cv2.polylines(self.img, [self._tri_pts(p1, p2)], True, col, thick)

    def fill_triangle(self, p1, p2, col):
        if not (p1 and p2): return
        cv2.fillPoly(self.img, [self._tri_pts(p1, p2)], col)

    def put_text(self,text,pos,col):
        cv2.putText(self.img,text,pos,cv2.FONT_HERSHEY_SIMPLEX,1.4,col,3)

    def erase(self,pt,size):
        if pt: cv2.circle(self.img,pt,size//2,(0,0,0),-1)

    # ── Selection / Move ──
    def grab_region(self, x1,y1,x2,y2):
        """Cut out a rectangle from the canvas into sel_patch."""
        x1,x2=sorted([max(x1,0),min(x2,self.w)])
        y1,y2=sorted([max(y1,0),min(y2,self.h)])
        if x2<=x1 or y2<=y1: return False
        self.sel_patch=self.img[y1:y2,x1:x2].copy()
        self.sel_x=x1; self.sel_y=y1
        # Erase from canvas
        cv2.rectangle(self.img,(x1,y1),(x2,y2),(0,0,0),-1)
        return True

    def move_sel(self, dx, dy):
        if self.sel_patch is not None:
            self.sel_x+=dx; self.sel_y+=dy

    def drop_sel(self):
        if self.sel_patch is None: return
        ph,pw=self.sel_patch.shape[:2]
        x1=max(self.sel_x,0); y1=max(self.sel_y,0)
        x2=min(self.sel_x+pw,self.w); y2=min(self.sel_y+ph,self.h)
        if x2>x1 and y2>y1:
            sx=x1-self.sel_x; sy=y1-self.sel_y
            region=self.sel_patch[sy:sy+(y2-y1),sx:sx+(x2-x1)]
            mask=region.any(axis=2)
            self.img[y1:y2,x1:x2][mask]=region[mask]
        self.sel_patch=None

    def draw_sel_preview(self, frame):
        """Render floating selection onto the display frame."""
        if self.sel_patch is None: return
        ph,pw=self.sel_patch.shape[:2]
        x1=max(self.sel_x,0); y1=max(self.sel_y,0)
        x2=min(self.sel_x+pw,frame.shape[1]); y2=min(self.sel_y+ph,frame.shape[0])
        if x2>x1 and y2>y1:
            sx=x1-self.sel_x; sy=y1-self.sel_y
            region=self.sel_patch[sy:sy+(y2-y1),sx:sx+(x2-x1)]
            mask=region.any(axis=2)
            frame[y1:y2,x1:x2][mask]=region[mask]
        # Dashed border
        cv2.rectangle(frame,(self.sel_x,self.sel_y),
                      (self.sel_x+pw,self.sel_y+ph),(0,255,255),2)

    def undo(self):
        self.drop_sel()
        if self.history: self.img=self.history.pop(); return True
        return False

    def clear(self):
        self.snapshot(); self.img[:]=0; self.drop_sel()

    def blend(self, frame, bg_white):
        bg = np.full_like(frame,255) if bg_white else frame.copy()
        mask=self.img.any(axis=2)
        out=bg.copy(); out[mask]=self.img[mask]
        return out

    def save(self, directory, bg_white, frame):
        self.drop_sel()
        os.makedirs(directory,exist_ok=True)
        ts=datetime.now().strftime("%Y%m%d_%H%M%S")
        path=os.path.join(directory,f"drawing_{ts}.png")
        cv2.imwrite(path,self.blend(frame,bg_white))
        return path

# ── TEXT OVERLAY ────────────────────────────────
def draw_text_overlay(frame, typed):
    h,w=frame.shape[:2]
    ov=frame.copy()
    cv2.rectangle(ov,(w//2-340,h//2-65),(w//2+340,h//2+65),(15,15,15),-1)
    cv2.addWeighted(ov,0.8,frame,0.2,0,frame)
    cv2.rectangle(frame,(w//2-340,h//2-65),(w//2+340,h//2+65),(0,200,255),2)
    cv2.putText(frame,"TYPE TEXT — ENTER to confirm, ESC to cancel",
                (w//2-310,h//2-28),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,200,255),1)
    cv2.putText(frame,typed+"|",
                (w//2-310,h//2+25),cv2.FONT_HERSHEY_SIMPLEX,1.1,(255,255,255),2)

# ── MAIN ────────────────────────────────────────
def main():
    cap=cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    ret,frame=cap.read()
    if not ret: print("❌ Cannot open camera."); return
    h,w=frame.shape[:2]

    det    =HandDetector()
    toolbar=Toolbar(w)
    canvas =Canvas(h,w)

    active_idx  =0; tool='FreeDraw'; brush=8; bg_white=False
    prev_pt     =None; shape_start=None
    mode_label  ="Idle"; erase_cur=None
    status_msg  =""; status_timer=0
    last_frame  =frame

    # Hover state for toolbar
    hov_item=None; hov_start=None; hov_prog=0.0

    # Text state
    text_mode=False; typed_text=""; text_ready=False; pending_text=""

    # Selection/move state
    sel_start=None          # first corner of selection box
    sel_dragging=False      # currently dragging selection
    sel_drag_prev=None      # previous drag point

    print("✅ Air Drawing System — Complete Edition!")
    print("1=Draw 2=Line 3=Circle 4=Rect 5=Triangle 6=Text 7=Select | B Z C S Q")
    print("✌️ Hover over + or − buttons in toolbar to resize brush")

    while True:
        ret,frame=cap.read()
        if not ret: break
        frame=cv2.flip(frame,1); last_frame=frame.copy()

        frame  =det.process(frame)
        pos    =det.landmarks(frame)
        fingers=det.fingers_up(pos)
        fist   =det.is_fist(fingers)
        index_tip=pos[8] if len(pos)>=21 else None

        hov_prog=0.0

        # ══════════════════════════════════
        #  TEXT TYPING OVERLAY
        # ══════════════════════════════════
        if text_mode:
            draw_text_overlay(frame,typed_text)
            blended=canvas.blend(frame,bg_white)
            canvas.draw_sel_preview(blended)
            toolbar.draw(blended,active_idx,tool,brush,bg_white)
            cv2.imshow("Air Drawing System",blended)
            key=cv2.waitKey(1)&0xFF
            if key==13:
                if typed_text.strip():
                    pending_text=typed_text.strip(); text_ready=True
                    status_msg=f"☝️ Point to place: '{pending_text}'"; status_timer=200
                text_mode=False; typed_text=""
            elif key==27: text_mode=False; typed_text=""
            elif key in(8,127): typed_text=typed_text[:-1]
            elif 32<=key<=126: typed_text+=chr(key)
            continue

        # ══════════════════════════════════
        #  TEXT PLACEMENT
        # ══════════════════════════════════
        if text_ready:
            mode_label=f"☝️ Point to place: '{pending_text}'"
            if index_tip and index_tip[1]>HEADER_HEIGHT:
                cv2.putText(frame,pending_text,index_tip,
                            cv2.FONT_HERSHEY_SIMPLEX,1.4,COLORS[active_idx][1],3)
                if fist:
                    canvas.snapshot()
                    canvas.put_text(pending_text,index_tip,COLORS[active_idx][1])
                    text_ready=False; pending_text=""
                    status_msg,status_timer="✅ Text placed!",60
            blended=canvas.blend(frame,bg_white)
            canvas.draw_sel_preview(blended)
            toolbar.draw(blended,active_idx,tool,brush,bg_white)
            cv2.putText(blended,"✊ FIST to stamp",(w//2-120,h-20),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
            if status_timer>0:
                cv2.putText(blended,status_msg,(w//2-200,h-55),
                            cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,255,100),2)
                status_timer-=1
            cv2.imshow("Air Drawing System",blended)
            key=cv2.waitKey(1)&0xFF
            if key==ord('q'): break
            elif key==27: text_ready=False; pending_text=""
            continue

        # ══════════════════════════════════
        #  HOVER MODE  ✌️ (color, +/-, tools)
        # ══════════════════════════════════
        if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
            mode_label="✌️ Hover over toolbar item"
            prev_pt=None; shape_start=None; canvas.end_stroke()

            if index_tip:
                item=toolbar.hit_test(*index_tip)
                if item:
                    if item!=hov_item:
                        hov_item=item; hov_start=time.time()
                    elapsed=time.time()-hov_start
                    hov_prog=min(elapsed/HOVER_TIME,1.0)

                    # Activate after hover time
                    if elapsed>=HOVER_TIME:
                        hov_item=None; hov_start=None
                        if item[0]=='color':
                            active_idx=min(item[1],len(COLORS)-1)
                            status_msg=f"✅ {COLORS[active_idx][0]}"; status_timer=60
                        elif item[0]=='plus':
                            brush=min(brush+4,MAX_BRUSH)
                            status_msg=f"Brush: {brush}px"; status_timer=40
                        elif item[0]=='minus':
                            brush=max(brush-4,MIN_BRUSH)
                            status_msg=f"Brush: {brush}px"; status_timer=40
                        elif item[0]=='tool':
                            t=TOOLS.get(item[1])
                            if t:
                                if t=='Text': text_mode=True; typed_text=""
                                else: tool=t; shape_start=None
                                status_msg=f"Tool: {t}"; status_timer=50
                else:
                    hov_item=None; hov_start=None
            else:
                hov_item=None; hov_start=None

        # ══════════════════════════════════
        #  SELECT & MOVE TOOL  (tool=Select)
        # ══════════════════════════════════
        elif tool=='Select':
            mode_label="7:Select — ☝️ drag box → ✊ grab → move → ✊ drop"

            # If we already have a floating selection, dragging moves it
            if canvas.sel_patch is not None:
                mode_label="☝️ Move selection — ✊ to drop"
                if index_tip and index_tip[1]>HEADER_HEIGHT:
                    if sel_drag_prev is not None:
                        dx = index_tip[0] - sel_drag_prev[0]
                        dy = index_tip[1] - sel_drag_prev[1]
                        canvas.move_sel(dx, dy)
                    sel_drag_prev = index_tip
                if fist:
                    canvas.snapshot()
                    canvas.drop_sel()
                    sel_start=None; sel_drag_prev=None
                    status_msg,status_timer="📌 Dropped!",60

            else:
                # Drawing the selection rectangle with index finger
                if fingers[1] and not fingers[2]:
                    if index_tip and index_tip[1]>HEADER_HEIGHT:
                        if sel_start is None:
                            sel_start=index_tip
                        # Show live selection box on frame
                        cv2.rectangle(frame,sel_start,index_tip,(0,255,255),2)
                        # Dashes at corners
                        for pt in [sel_start,index_tip,(sel_start[0],index_tip[1]),(index_tip[0],sel_start[1])]:
                            cv2.circle(frame,pt,5,(0,255,255),-1)
                        cv2.putText(frame,"✊ FIST to grab selected area",
                                    (10,h-20),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,255,255),2)
                    else:
                        sel_start=None

                # Fist = grab the drawn selection
                if fist and sel_start and index_tip:
                    x1,x2=sorted([sel_start[0],index_tip[0]])
                    y1,y2=sorted([sel_start[1],index_tip[1]])
                    if x2-x1>10 and y2-y1>10:
                        canvas.snapshot()
                        ok=canvas.grab_region(x1,y1,x2,y2)
                        if ok:
                            sel_drag_prev=index_tip
                            status_msg,status_timer="✅ Grabbed! Move hand to drag",70
                        else:
                            status_msg,status_timer="⚠️ Nothing selected",40
                    sel_start=None

                if not fingers[1]:
                    sel_start=None

        # ══════════════════════════════════
        #  FIST = commit shape / end stroke
        # ══════════════════════════════════
        elif fist:
            mode_label="✊ Pen lifted"
            color=COLORS[active_idx][1]
            if shape_start and index_tip and tool not in ('FreeDraw','Eraser','Text','Select'):
                canvas.snapshot()
                if   tool=='Line':      canvas.draw_line(shape_start,index_tip,color,brush)
                elif tool=='Circle':    canvas.draw_circle(shape_start,index_tip,color,brush)
                elif tool=='Rectangle': canvas.draw_rect(shape_start,index_tip,color,brush)
                elif tool=='Triangle':  canvas.draw_triangle(shape_start,index_tip,color,brush)
                status_msg,status_timer=f"✅ {tool} drawn!",55
            shape_start=None; prev_pt=None; hov_item=None; canvas.end_stroke()

        # ══════════════════════════════════
        #  DRAW  ☝️
        # ══════════════════════════════════
        elif fingers[1] and not fingers[2]:
            is_eraser=(active_idx==len(COLORS)-1)
            color=COLORS[active_idx][1]
            hov_item=None
            mode_label="Erasing" if is_eraser else f"☝️ {tool} — {COLORS[active_idx][0]}"

            if index_tip and index_tip[1]>HEADER_HEIGHT:
                if is_eraser:
                    canvas.snapshot(); canvas.erase(index_tip,ERASER_SIZE)
                    erase_cur=index_tip; prev_pt=None

                elif tool=='FreeDraw':
                    canvas.snapshot()
                    canvas.draw_line(prev_pt,index_tip,color,brush)
                    prev_pt=index_tip
                    cv2.circle(frame,index_tip,max(1,brush//2),color,2)

                elif tool in ('Line','Circle','Rectangle','Triangle'):
                    if shape_start is None: shape_start=index_tip
                    # Live preview on frame only
                    if tool=='Line':
                        cv2.line(frame,shape_start,index_tip,color,max(brush,2))
                    elif tool=='Circle':
                        r=int(math.hypot(index_tip[0]-shape_start[0],index_tip[1]-shape_start[1]))
                        if r>0: cv2.circle(frame,shape_start,r,color,max(brush,2))
                    elif tool=='Rectangle':
                        cv2.rectangle(frame,shape_start,index_tip,color,max(brush,2))
                    elif tool=='Triangle':
                        ax,ay=shape_start; bx,by=index_tip
                        cx=int(2*ax-bx); cy=by
                        pts=np.array([[ax,ay],[bx,by],[cx,cy]],np.int32)
                        cv2.polylines(frame,[pts],True,color,max(brush,2))
                    cv2.circle(frame,shape_start,7,(255,255,0),-1)
                    cv2.putText(frame,"✊ FIST to stamp shape",
                                (10,h-20),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,255,255),2)
            else:
                prev_pt=None; canvas.end_stroke()

        # ══════════════════════════════════
        #  IDLE
        # ══════════════════════════════════
        else:
            prev_pt=None; shape_start=None
            hov_item=None; hov_start=None; canvas.end_stroke()
            mode_label="Idle — ✌️ toolbar | ☝️ draw | ✊ end/drop"

        # ══════════════════════════════════
        #  RENDER
        # ══════════════════════════════════
        blended=canvas.blend(frame,bg_white)
        canvas.draw_sel_preview(blended)   # floating selection on top

        if mode_label=="Erasing" and erase_cur:
            cv2.circle(blended,erase_cur,ERASER_SIZE//2,(160,160,160),2)

        toolbar.draw(blended,active_idx,tool,brush,bg_white,
                     hov_item if hov_start else None, hov_prog)

        if status_timer>0:
            cv2.putText(blended,status_msg,(w//2-200,h-20),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,100),2)
            status_timer-=1

        # Color indicator bottom-left
        lbl,bgr=COLORS[active_idx]
        cv2.rectangle(blended,(10,h-65),(65,h-10),bgr,-1)
        cv2.rectangle(blended,(10,h-65),(65,h-10),(255,255,255),2)
        cv2.putText(blended,lbl,(72,h-26),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)

        # Mode label
        cv2.putText(blended,mode_label,(8,HEADER_HEIGHT+46),
                    cv2.FONT_HERSHEY_SIMPLEX,0.48,(80,255,80),1)

        cv2.imshow("Air Drawing System",blended)

        key=cv2.waitKey(1)&0xFF
        if key==ord('q'): break
        elif key==ord('c'): canvas.clear(); status_msg,status_timer="Canvas Cleared!",60
        elif key==ord('s'):
            path=canvas.save(SAVE_DIR,bg_white,last_frame)
            status_msg,status_timer=f"Saved: {os.path.basename(path)}",90
            print(f"💾 {path}")
        elif key==ord('z'):
            ok=canvas.undo()
            status_msg,status_timer=("↩️ Undo",50) if ok else ("Nothing to undo",40)
        elif key==ord('b'):
            bg_white=not bg_white
            status_msg=("⬜ White BG" if bg_white else "📷 Camera BG"); status_timer=60
        elif key in(ord('1'),ord('2'),ord('3'),ord('4'),ord('5')):
            tool=TOOLS[chr(key)]; shape_start=None; canvas.drop_sel()
            status_msg,status_timer=f"Tool: {TOOLS[chr(key)]}",50
        elif key==ord('6'): tool='Text'; text_mode=True; typed_text=""
        elif key==ord('7'):
            tool='Select'; canvas.drop_sel()
            status_msg,status_timer="Select tool — ☝️ draw box ✊ grab",60
        elif key in(ord('='),ord('+')):
            brush=min(brush+2,MAX_BRUSH); status_msg,status_timer=f"Brush: {brush}px",30
        elif key==ord('-'):
            brush=max(brush-2,MIN_BRUSH); status_msg,status_timer=f"Brush: {brush}px",30

    cap.release(); cv2.destroyAllWindows(); print("👋 Goodbye!")

if __name__=="__main__":
    main()
