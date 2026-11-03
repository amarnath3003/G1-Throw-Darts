"""Image-supported pen centerline hypotheses; inferred spans never become contacts."""
import cv2
import numpy as np


def track_pen(image, hand_box=None, previous=None, table=None):
    """Link collinear blue-white pen fragments near the hand at half resolution.

    The pen is pale blue against yellow wood. Color is an appearance prior for
    this dataset, not a generic object tracker. Endpoints remain visibility-
    dependent and cap identity is not automatically observable.
    """
    small=cv2.resize(image,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
    h,w=small.shape[:2];b,g,r=cv2.split(small.astype(np.float32))
    mask=((b>110)&(b>r*1.015)&(g>r*.98)&(g>100)).astype(np.uint8)*255
    mask[:int(h*.08)]=0;mask[int(h*.50):]=0;mask[:,:int(w*.25)]=0
    if hand_box is not None:
        box=np.array(hand_box)*.5
        xx,yy=np.meshgrid(np.arange(w),np.arange(h))
        # Generous box expansion retains pen tips outside the hand detector.
        near=(xx>=box[0]-150)&(xx<=box[2]+150)&(yy>=box[1]-150)&(yy<=box[3]+150)
        mask[~near]=0
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((2,2),np.uint8))
    lines=cv2.HoughLinesP(mask,1,np.pi/360,threshold=22,minLineLength=25,maxLineGap=55)
    if lines is None:return dict(valid=False,reason='no_pen_line_evidence')
    ys,xs=np.nonzero(mask);pixels=np.column_stack([xs,ys]).astype(float)
    candidates=[]
    # OpenCV builds differ between (N,1,4) and (N,4) return layouts.
    for line in np.asarray(lines).reshape(-1,4)[:100]:
        a,bp=line[:2].astype(float),line[2:].astype(float);axis=bp-a;axis/=max(np.linalg.norm(axis),1e-8)
        perp=np.array([-axis[1],axis[0]])
        close=pixels[np.abs((pixels-a)@perp)<5]
        if len(close)<100:continue
        projection=(close-a)@axis
        # Limit to the local line: do not join unrelated page/furniture edges.
        local=(projection>min(0,np.dot(bp-a,axis))-90)&(projection<max(0,np.dot(bp-a,axis))+90)
        close=close[local];projection=projection[local]
        if len(close)<100:continue
        low,high=np.percentile(projection,[1,99]);length=high-low
        if not 90<length<260:continue
        bins=np.histogram(projection,bins=np.arange(low,high+5,5))[0]
        coverage=float(np.mean(bins>=4))
        if coverage<.52:continue
        center=a+axis*((low+high)/2)+perp*np.median((close-a)@perp)
        endpoints=np.array([center-axis*length/2,center+axis*length/2])*2
        if previous is not None and np.linalg.norm(endpoints-previous[::-1])<np.linalg.norm(endpoints-previous):endpoints=endpoints[::-1]
        if previous is None and endpoints[0,1]>endpoints[1,1]:endpoints=endpoints[::-1]
        if table is not None:
            points=cv2.perspectiveTransform(endpoints.astype(np.float32)[None],np.array(table['image_to_board']))[0]
            projected_length=float(np.linalg.norm(points[1]-points[0]))
            if not .065<projected_length<.23:continue
        else:projected_length=None
        continuity=0 if previous is None else min(np.linalg.norm(endpoints.mean(0)-previous.mean(0))/400,1)
        score=coverage+min(len(close)/800,1)*.25-.2*continuity
        candidates.append((score,endpoints,coverage,projected_length))
    if not candidates:return dict(valid=False,reason='insufficient_collinear_pen_support')
    score,endpoints,coverage,plane_length=max(candidates,key=lambda x:x[0])
    return dict(valid=True,endpoints_px=endpoints.tolist(),observed_line_coverage=coverage,
                confidence='image_supported_partial_line' if coverage<.9 else 'image_supported_line',
                projected_table_length_m=plane_length,cap_identity='unverified',
                hidden_span_inferred=coverage<.9,contact_observed=False,
                limitations='Endpoints can truncate at occlusion. Airborne plane projection is not actual pen length.')
