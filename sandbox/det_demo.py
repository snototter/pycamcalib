import cv2
import numpy as np
from vito import imvis, imutils

# from dataclasses import dataclass

# from svglib.svglib import svg2rlg
# from reportlab.graphics import renderPDF, renderPM
import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')))
from pcc import patterns


def load_images():
    return [imutils.imread('calib/test{:d}.jpg'.format(i)) for i in [0, 1, 6]]#range(6)]

def hough_lines(img):
    g = imutils.grayscale(img)
    vis = img.copy()
    edges = cv2.Canny(g,100,200,apertureSize = 3)
    #lines = cv2.HoughLines(edges,1,np.pi/180,50)
    # for rho,theta in lines[0]:
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a*rho
    #     y0 = b*rho
    #     x1 = int(x0 + 1000*(-b))
    #     y1 = int(y0 + 1000*(a))
    #     x2 = int(x0 - 1000*(-b))
    #     y2 = int(y0 - 1000*(a))
    #     cv2.line(vis,(x1,y1),(x2,y2),(255,0,255),2)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, minLineLength=50, maxLineGap=5)
    if lines is not None:
        for r in range(lines.shape[0]):
            # print(lines[r,:].shape, lines[r,:][0], lines[r,0,0], lines[r,:,:], lines[r,:])
            x1, y1, x2, y2 = lines[r,:][0]
        # for x1, y1, x2, y2 in lines:
            cv2.line(vis,(x1,y1),(x2,y2),(255,0,255),2)
        #imvis.imshow(edges, title='edges', wait_ms=10)
    imvis.imshow(vis, title='Hough', wait_ms=10)
    

def contour(img):
    g = imutils.grayscale(img)
    _, g = cv2.threshold(g, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # g = imutils.gaussian_blur(g, 1)
    # g = cv2.blur(g, (3,3))
    vis = imutils.ensure_c3(g)
    vis = img.copy()
    edges = cv2.Canny(g,50,200,apertureSize = 3)
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges,kernel,iterations = 1)
    cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    #https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
    # White on black!

    approximated_polygons = list()
    for cnt in cnts:
        # get simplified convex hull
        epsilon = 0.05*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        hull = cv2.convexHull(approx)

        # cv2.drawContours(vis, [approx], 0, (0,255,0), 3)
        approximated_polygons.append({'hull': hull, 'approx': approx, 'hull_area': cv2.contourArea(hull),
        'corners': len(hull)
        })
    
    def _key(a):
        return a['hull_area']
    approximated_polygons.sort(key=_key, reverse=True)
    i = 0
    for approx in approximated_polygons:
        #cv2.drawContours(vis, [approx['cnt']], 0, (0,255,0) if i < 10 else (255,0,0), 3)
        cv2.drawContours(vis, [approx['hull']], 0, (0,255,0) if 3 < approx['corners'] < 6 else (255,0,0), 3)
        i += 1
        # if i < 15:
            # print('Largest', i, approx['approx'], approx['area'])
            # imvis.imshow(vis, title='contours', wait_ms=-1)


    imvis.imshow(vis, title='contours', wait_ms=-1)

#https://stackoverflow.com/questions/57767518/detecting-white-blobs-on-grayscale-image-using-mser-in-opencv
def mser(img):
    g = imutils.grayscale(img)
    vis = img.copy()
    mser = cv2.MSER_create(_min_area=1000)
    regions, _ = mser.detectRegions(g)
    for p in regions:
        xmax, ymax = np.amax(p, axis=0)
        xmin, ymin = np.amin(p, axis=0)
        cv2.rectangle(vis, (xmin,ymax), (xmax,ymin), (0, 255, 0), 1)
    imvis.imshow(vis, title='mser', wait_ms=-1)
    # vis = img.copy()
    # ms = cv2.MSER()
    # m_areas = ms.detect(g)



def fld_lines(img):
    #FIXME requires opencv-contrib-python
    g = imutils.grayscale(img)
    fld = cv2.ximgproc.createFastLineDetector()
    lines = fld.detect(g)
    vis = img.copy()
    vis = fld.drawSegments(vis, lines)
    imvis.imshow(vis, title='FLD lines', wait_ms=10)


def det_demo(imgs):
    for img in imgs:
        # fld_lines(img)
        # hough_lines(img)
        contour(img)
        # mser(img)


if __name__ == '__main__':
    pattern_specs = patterns.eddie.PatternSpecificationEddie('dummy',
        target_width_mm=210, target_height_mm=297,
        dia_circles_mm=5, dist_circles_mm=11, bg_color='orange')
    # patterns.eddie.ContourDetectionParams().get_marker_template(pattern_specs)

    # Match template:
    # #https://docs.opencv.org/master/df/dfb/group__imgproc__object.html#ga586ebfb0a7fb604b35a23d85391329be

    imgs = load_images()
    for img in imgs:
        patterns.eddie.find_marker(img, pattern_specs)
    # det_demo(imgs)



##### Stuff, that didn't work:
    # setting SVG clippath
    # #https://gist.github.com/jrief/25962a9194ec1c6c3c590bf8977fd0ff
    # drawing = svg2rlg("foo.svg")
    # left, bottom, right, top =  drawing.getBounds() # this returns negative bounds for our exported svgs!?!?
    # print('BOUNDS', drawing.getBounds())
    # from reportlab.graphics import renderSVG
    # canvas = renderSVG.SVGCanvas()
    # renderSVG.draw(drawing, canvas)
    # clipRect = next(iter(canvas.svg.getElementsByTagName('clipPath'))).firstChild
    # clipRect.setAttribute('x', f'{mrect.left*pattern.target_width_mm}mm')
    # clipRect.setAttribute('y', f'{mrect.top*pattern.target_height_mm}mm')
    # clipRect.setAttribute('width', f'{mrect.width*pattern.target_width_mm}mm')
    # clipRect.setAttribute('height', f'{mrect.height*pattern.target_height_mm}mm')
    # canvas.save('cropped.svg')

    # setting viewbox test
    # patterns.export_pattern(patterns.eddie.eddie_test_specs_a4, 'test-folder', None,
    #                       export_pdf=True, export_png=False,
    #                       prevent_overwrite=False)
    #Adjusting the viewbox didn't work well when exporting to PNG ("weird" scaling issues....)
    # dwg = pattern.render_svg()
    # print('\n'.join(['{}:{}'.format(k, dwg.attribs[k]) for k in dwg.attribs]))
    # dwg.attribs['viewBox'] = '0 0 200 290'
    # dwg.saveas('foo.svg')
    # drawing = svg2rlg('foo.svg')
    # renderPM.drawToFile(drawing, 'foo.png', fmt="PNG", dpi=72)