"""
hand_tryon.py
=============
Production-Level Ring & Bracelet Virtual Try-On — stabilized + sharpened
"""

import math
import cv2
import mediapipe as mp
import numpy as np

from overlay_utils import alpha_blend, alpha_blend_at
from transform_utils import (
    calculate_distance, unit_vector_2d,
    resize_to_width, rotate_image,
    build_perspective_quad,
    build_wrist_perspective_quad,
    perspective_warp,
    hand_occlusion_mask,
    apply_occlusion,
    JewelrySmoother,
)


# -------------------------------------------------
# Motion Stabilizer
# -------------------------------------------------

class MotionStabilizer:

    def __init__(self):
        self.prev = None

    def stabilize(self, value, slow=0.85, fast=0.35):

        if self.prev is None:
            self.prev = value
            return value

        delta = abs(value - self.prev)

        alpha = fast if delta > 15 else slow

        smoothed = alpha * self.prev + (1 - alpha) * value

        self.prev = smoothed
        return smoothed


# -------------------------------------------------
# Finger config
# -------------------------------------------------

FINGER_CONFIG = {
    "Thumb":  (2,3,4,0,5,0.40,[1,2,3,4]),
    "Index":  (5,6,7,2,9,0.50,[5,6,7,8,9]),
    "Middle": (9,10,11,5,13,0.50,[5,9,10,11,12,13]),
    "Ring":   (13,14,15,9,17,0.55,[9,13,14,15,16,17]),
    "Pinky":  (17,18,19,13,17,0.45,[13,17,18,19,20]),
}

_ALPHA = 0.68


# -------------------------------------------------
# 3D width projection
# -------------------------------------------------

def project_width_3d(center_3d, p_left_3d, p_right_3d, axis_unit_2d):

    perp = np.array([-axis_unit_2d[1], axis_unit_2d[0]])

    c2 = np.asarray(center_3d,float)[:2]

    proj_l = float(np.dot(np.asarray(p_left_3d,float)[:2]-c2, perp))
    proj_r = float(np.dot(np.asarray(p_right_3d,float)[:2]-c2, perp))

    return abs(proj_l - proj_r)


# -------------------------------------------------
# Bracelet curvature warp
# -------------------------------------------------

def trapezoid_warp(img, taper=0.10):

    if img is None:
        return img

    h,w = img.shape[:2]

    dx = int(w*taper)

    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst = np.float32([[dx,0],[w-dx,0],[w,h],[0,h]])

    M = cv2.getPerspectiveTransform(src,dst)

    return cv2.warpPerspective(
        img,M,(w,h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0,0,0,0)
    )


# -------------------------------------------------
# Main class
# -------------------------------------------------

class HandTryOn:

    WRIST=0
    INDEX_MCP=5
    MIDDLE_MCP=9
    RING_MCP=13
    PINKY_MCP=17
    THUMB_MCP=2

    def __init__(self):

        mp_h = mp.solutions.hands

        self.hands = mp_h.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75,
        )

        self.ring_sm = {f: JewelrySmoother(_ALPHA) for f in FINGER_CONFIG}
        self.bracelet_sm = JewelrySmoother(_ALPHA)

        self.cx_stab = MotionStabilizer()
        self.cy_stab = MotionStabilizer()
        self.scale_stab = MotionStabilizer()


    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def process(self, frame, jewelry_img, jewelry_type,
                finger="Ring",
                ring_scale=1.0,
                bracelet_scale=1.0):

        h,w = frame.shape[:2]

        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        res = self.hands.process(rgb)

        if (not res
            or not res.multi_hand_landmarks
            or not res.multi_hand_world_landmarks):
            return frame

        slm = res.multi_hand_landmarks[0].landmark
        wlm = res.multi_hand_world_landmarks[0].landmark

        if jewelry_type=="ring":

            return self.apply_ring(
                frame,slm,wlm,jewelry_img,w,h,finger,ring_scale
            )

        if jewelry_type=="bracelet":

            return self.apply_bracelet(
                frame,slm,wlm,jewelry_img,w,h,bracelet_scale
            )

        return frame


    # -------------------------------------------------
    # Landmark helpers
    # -------------------------------------------------

    def spx(self,lm,i,w,h):
        l=lm[i]
        return np.array([l.x*w,l.y*h])

    def wpx(self,lm,i):
        l=lm[i]
        return np.array([l.x,l.y,l.z])


    # -------------------------------------------------
    # Ring
    # -------------------------------------------------

    def apply_ring(self,frame,slm,wlm,img,w,h,finger,scale_mul):

        cfg = FINGER_CONFIG[finger]

        mcp,pip,dip,l_i,r_i,wf,occ_hull = cfg

        s_mcp = self.spx(slm,mcp,w,h)
        s_pip = self.spx(slm,pip,w,h)

        seg_vec = s_pip - s_mcp
        seg_len = np.linalg.norm(seg_vec)

        if seg_len<2:
            return frame

        w_mcp = self.wpx(wlm,mcp)
        w_pip = self.wpx(wlm,pip)
        w_dip = self.wpx(wlm,dip)

        w_l = self.wpx(wlm,l_i)
        w_r = self.wpx(wlm,r_i)

        world_dir = w_dip - w_mcp
        world_axis = world_dir[:2]/(np.linalg.norm(world_dir[:2])+1e-9)

        cross_dist = project_width_3d(w_mcp,w_l,w_r,world_axis)

        world_seg = np.linalg.norm(w_pip-w_mcp)+1e-9

        px_per_m = seg_len/world_seg

        finger_w = cross_dist*wf*px_per_m

        depth = float(w_mcp[2])

        depth_fac = max(0.6,min(1.5,1.0-depth*5.5))

        ring_w_raw = max(finger_w*depth_fac*scale_mul,8)

        t=0.45

        cx_raw = float(s_mcp[0]+t*seg_vec[0])
        cy_raw = float(s_mcp[1]+t*seg_vec[1])

        screen_dir = unit_vector_2d(s_mcp,s_pip)

        angle_raw = float(np.degrees(np.arctan2(
            screen_dir[1],screen_dir[0])))+90

        sm = self.ring_sm[finger].smooth(
            cx=cx_raw,cy=cy_raw,scale=ring_w_raw,angle=angle_raw
        )

        cx = self.cx_stab.stabilize(float(sm["cx"]))
        cy = self.cy_stab.stabilize(float(sm["cy"]))

        rw = max(int(self.scale_stab.stabilize(sm["scale"])),8)

        angle = float(sm["angle"])

        axis = unit_vector_2d(s_mcp,s_pip)

        rh = int(rw*img.shape[0]/max(img.shape[1],1))

        quad = build_perspective_quad(cx,cy,rw,rh,axis)

        original = frame.copy()

        result = perspective_warp(img,quad)

        if result is not None:

            warped,origin = result

            frame = alpha_blend_at(frame,warped,origin[0],origin[1])

        else:

            proc = resize_to_width(img,rw)

            proc = rotate_image(proc,angle)

            frame = alpha_blend(frame,proc,cx,cy)

        try:

            mask = hand_occlusion_mask(frame.shape,slm,w,h,4)

            frame = apply_occlusion(
                frame,original,mask,
                occ_hull,slm,w,h,
                occlusion_strength=0.4
            )

        except:
            pass

        return frame


    # -------------------------------------------------
    # Bracelet
    # -------------------------------------------------

    def apply_bracelet(self,frame,slm,wlm,img,w,h,scale_mul):

        s_wrist = self.spx(slm,self.WRIST,w,h)
        s_idx = self.spx(slm,self.INDEX_MCP,w,h)
        s_pink = self.spx(slm,self.PINKY_MCP,w,h)

        knuckle_span = calculate_distance(s_idx,s_pink)

        depth = float(self.wpx(wlm,self.WRIST)[2])

        depth_fac=max(0.6,min(1.5,1.0-depth*4.5))

        bracelet_w = max(knuckle_span*0.75*depth_fac*scale_mul,12)

        cx=s_wrist[0]
        cy=s_wrist[1]

        wrist_axis = unit_vector_2d(s_idx,s_pink)

        angle=np.degrees(np.arctan2(wrist_axis[1],wrist_axis[0]))

        sm=self.bracelet_sm.smooth(cx=cx,cy=cy,scale=bracelet_w,angle=angle)

        bw=max(int(sm["scale"]),12)

        proc=resize_to_width(img,bw)

        proc=trapezoid_warp(proc)

        bh=int(bw*proc.shape[0]/max(proc.shape[1],1))

        quad=build_wrist_perspective_quad(s_idx,s_pink,s_wrist,bw,bh,depth_z=depth)

        result=perspective_warp(proc,quad)

        if result is not None:

            warped,origin=result

            frame=alpha_blend_at(frame,warped,origin[0],origin[1])

        else:

            rot=rotate_image(proc,angle)

            frame=alpha_blend(frame,rot,cx,cy)

        return frame