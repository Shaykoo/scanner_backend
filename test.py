import cv2
import numpy as np
import math

def scan(corDet):
    org = corDet.copy()
    org1 = corDet.copy()
    height, width, channel = corDet.shape
    grey  = cv2.cvtColor(corDet, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(grey, 255, 255)
    ret, thresh = cv2.threshold(edge, 127, 255, cv2.THRESH_BINARY)
    cont, cod = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    Dcont = cv2.drawContours(org1, cont, -1, (0,0,255),3)
    corLis=[]
    for i in cont:
	    area = cv2.contourArea(i)
	    if area>500:
	    	perimeter = cv2.arcLength(i, True)
	    	cor = cv2.approxPolyDP(i, 0.02*perimeter, True)
	    	if len(cor)==4:
	    		corLis = cor
    # print(corLis)

    dis = []
    for i in corLis:
    	x, y = i.ravel()
    	d = math.dist((0,0), (x,y))
    	xy = "("+str(x)+","+str(y)+")"
    	cv2.putText(corDet, xy, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255))
    	dis.append([x,y,int(d)])
    	dis.sort(key= lambda x : x[2])
    if dis[2][0]>dis[2][1]:
    	dis[1],dis[2] = dis[2],dis[1]
    Owidth = 0
    Oheight = 0
    w1 = math.dist((dis[0][0],dis[0][1]), (dis[1][0],dis[1][1]))
    w2 = math.dist((dis[2][0],dis[2][1]), (dis[3][0],dis[3][1]))
    h1 = math.dist((dis[0][0],dis[0][1]), (dis[2][0],dis[2][1]))
    h2 = math.dist((dis[1][0],dis[1][1]), (dis[3][0],dis[3][1]))
    if w1>w2:
    	Owidth = w1
    else:
    	Owidth = w2
    if h1>h2:
    	Oheight = h1
    else:
    	Oheight = h2
    # print(int(Owidth), int(Oheight))

    for i in corLis:
	    x,y = i.ravel()
	    corDet = cv2.circle(corDet, (x,y), 5, (0,0,255),cv2.FILLED)
    pts1 = np.float32(corLis)
    pts2 = np.float32([[0,0],[0,height],[width,height],[width,0]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    wap = cv2.warpPerspective(org, matrix, (width,height))
    out = cv2.resize(wap, (int(Owidth),int(Oheight)))
    return [org, grey, edge, Dcont, corDet, wap, out]

def display(Pimg,p=1,s=1):
	org = Pimg[0]
	grey = cv2.cvtColor(Pimg[1],cv2.COLOR_GRAY2BGR)
	edge = cv2.cvtColor(Pimg[2],cv2.COLOR_GRAY2BGR)
	Dcont = Pimg[3]
	corDet = Pimg[4]
	wap = Pimg[5]
	height, width, channel = wap.shape
	Oheight, Owidth, Ochanel = Pimg[6].shape
	meg1 = np.concatenate((org,grey,edge),1)
	meg2 = np.concatenate((Dcont,corDet,wap),1)
	meg3 = np.concatenate((meg1,meg2),0)
	Mheight, Mwidth, Mchannel = meg3.shape
	ip = cv2.resize(meg3, (int(Mwidth*p),int(Mheight*p)))
	io = cv2.resize(Pimg[6], (int(Owidth*s),int(Oheight*s)))
	return ip, io

if __name__=='__main__':
	img = cv2.imread("sample1.jpg") # enter your image path here
	ip, io = display(scan(img),0.5,1)
	cv2.imshow("process",ip)
	cv2.imshow("output", io)
	cv2.waitKey(0)
	# cv2.imwrite("output1.jpg",io)