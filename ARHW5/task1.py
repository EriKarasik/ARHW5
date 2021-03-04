import numpy as np
from numpy import sin, cos, radians, arctan2 as atan2, tan, arange, vstack as v, hstack as h
from numpy.linalg import inv, pinv
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.xlim(-1300, 1300)
plt.ylim(-1300, 1300)

Platform  =  30 #Platform
head  =  50 #Base
PJ = 800 #passive joint length
AJ =  300 #active joint length
#programm work too much time, so that is little time economy
cos30,sin30 = cos(radians(30)), sin(radians(30))
cos60, sin60 = cos(radians(60)), sin(radians(60))
cos120, sin120 = cos(radians(120)), sin(radians(120))
cos150, sin150 = cos(radians(150)), sin(radians(150))
cos270, sin270 = cos(radians(270)), sin(radians(270))
tan30, tan60 = tan(radians(30)), tan(radians(60))
R = head * 3 ** 0.5 / 3
r = Platform * 3 ** 0.5 / 3

def plotRobot(q,p):
    base = [[-R*cos30, R*cos30, 0,-R*cos30],[-R/2, -R/2, R, -R/2],[0,0,0,0]]
    platform = [[p[0]-r*cos30,p[0]+r*cos30,p[0],p[0]-r*cos30],[p[1]-r/2,p[1]-r/2,p[1]+r,p[1]-r/2],[p[2],p[2],p[2],p[2]]]
    activeJoint1 = [[cos30*R/2,cos30*R/2+AJ*sin(q[0])],[sin30*R/2,sin30*R/2+AJ*sin(q[0])],[0,AJ*cos(q[0])]]
    activeJoint2 = [[cos150*R/2,cos150*R/2+AJ*sin(q[1])],[sin150*R/2,sin150*R/2+AJ*sin(q[1])],[0,AJ*cos(q[1])]]
    activeJoint3 = [[cos270*R/2,cos270*R/2+AJ*sin(q[2])],[sin270*R/2,sin270*R/2+AJ*sin(q[2])],[0,AJ*cos(q[2])]]
    passiveJoint1 = [[activeJoint1[0][1],cos30*r/2],[activeJoint1[1][1],sin30*r/2],[activeJoint1[2][1],platform[2][0]]]
    passiveJoint2 = [[activeJoint2[0][1],cos150*r/2],[activeJoint2[1][1],sin150*r/2],[activeJoint2[2][1],platform[2][1]]]
    passiveJoint3 = [[activeJoint3[0][1],cos270*r/2],[activeJoint3[1][1],sin270*r/2],[activeJoint3[2][1],platform[2][2]]]
    ax.plot(base[0], base[1], base[2])
    ax.plot(activeJoint1[0], activeJoint1[1], activeJoint1[2])
    ax.plot(activeJoint2[0], activeJoint2[1], activeJoint2[2])
    ax.plot(activeJoint3[0], activeJoint3[1], activeJoint3[2])
    ax.plot(passiveJoint1[0], passiveJoint1[1], passiveJoint1[2])
    ax.plot(passiveJoint2[0], passiveJoint2[1], passiveJoint2[2])
    ax.plot(passiveJoint3[0], passiveJoint3[1], passiveJoint3[2])
    ax.plot(platform[0], platform[1], platform[2])

def IK(p):
    if p[2] == 0: return [np.NAN,np.NAN,np.NAN]
    th = [0,0,0]
    y1 = -tan30 * head * 0.5
    p[1] -= tan30 * Platform * 0.5
    x = [p[0],p[0]*cos120+p[1]*sin120,p[0]*cos120-p[1]*sin120]
    y = [p[1],p[1]*cos120-p[0]*sin120,p[1]*cos120+p[0]*sin120]
    # z = a + b*y
    for i in range(3):
        a = (x[i]**2+y[i]**2+p[2]**2+AJ**2-PJ**2-y1**2)/(2*p[2])
        b = (y1-y[i])/p[2]
        # discriminant
        d = -(a+b*y1)**2+AJ*(b**2 * AJ**2)
        if d < 0: th[i] = np.NAN
        else:
            yj = ((y1-a*b-d**0.5)/(b**2+1))  # choosing outer povar
            th[i] = atan2(-(a+b*yj),(y1-yj))
    return th

def FK(th):
    theta1, theta2, theta3 = radians(th[0]), radians(th[1]), radians(th[2])
    t = (head-Platform) * tan30 / 2.0
    y1 = -(t + AJ*cos(theta1))
    z1 = -AJ * sin(theta1)
    y2 = (t + AJ*cos(theta2)) * cos30
    x2 = (t + AJ*cos(theta2)) * sin30
    z2 = -AJ * sin(theta2)
    y3 = (t + AJ*cos(theta3)) * cos30
    x3 = (t + AJ*cos(theta3)) * sin30
    z3 = -AJ * sin(theta3)
    d = (y2-y1)*x3 - (y3-y1)*x2
    w1 = y1*y1 + z1*z1
    w2 = x2*x2 + y2*y2 + z2*z2
    w3 = x3*x3 + y3*y3 + z3*z3
    # x = (a1*z + b1)/dnm
    a1 = (z2-z1)*(y3-y1) - (z3-z1)*(y2-y1)
    b1= -( (w2-w1)*(y3-y1) - (w3-w1)*(y2-y1) ) / 2
    # y = (a2*z + b2)/dnm
    a2 = -(z2-z1)*x3 + (z3-z1)*x2
    b2 = ( (w2-w1)*x3 - (w3-w1)*x2) / 2
    # a*z^2 + b*z + c = 0
    a = a1*a1 + a2*a2 + d*d
    b = 2 * (a1*b1 + a2*(b2 - y1*d) - z1*d*d)
    c = (b2 - y1*d)*(b2 - y1*d) + b1*b1 + d*d*(z1*z1 - PJ*PJ)
    # discriminant
    d = b*b - 4*a*c
    if d < 0: return np.NAN
    z0 = -0.5*(b + d**0.5) / a
    return [(a1*z0+b1)/d,(a2*z0+b2)/d,z0]

a = head-2*Platform
b = Platform*3**0.5-3**0.5*head/2
c = Platform-0.5*head

def GetMap():
    xscatter,yscatter,zscatter,deflections = [],[],[],[]
    for x in np.arange(-1000, 1000, 100):
        for y in np.arange(-1000, 1000, 100):
            for z in np.arange(-1100, -100, 100):
                th = IK([x,y,z])
                J = np.dot(inv(np.diag([AJ*((y+a)*sin(th[0])-z*cos(th[0])),-AJ*((3**0.5*(x+b)+y+c)*sin(th[1])+2*z*cos(th[1])),
                    AJ*((3**0.5*(x-b)-y-c)*sin(th[2])-2*z*cos(th[2]))])),
                           v([h([x,y+a+AJ*cos(th[0]),z+AJ*sin(th[0])]),
                    h([2*(x+b)-3**0.5*AJ*cos(th[1]),2*(y+c)-AJ*cos(th[1]),2*(z+AJ*sin(th[1]))]),
                              h([2*(x-b)+3**0.5*AJ*cos(th[2]),2*(y+c)-AJ*cos(th[2]),2*(z+AJ*sin(th[2]))])]))
                if not np.isnan(np.sum(J)):
                    m = np.linalg.det(np.dot(J, np.transpose(J)))**0.5
                    xscatter.append(x)
                    yscatter.append(y)
                    zscatter.append(z)
                    deflections.append(m)
    plt.colorbar(ax.scatter3D(xscatter, yscatter, zscatter, c=deflections, cmap=plt.cm.get_cmap('viridis', 12), s=60))
    plt.show()
'''
def singularityMap():
    for x in arange(-10,10):
        for y in arange(-10, 10):
            for z in arange(-11, 0):
                if np.isnan(np.sum(IK([100*x,100*y,100*z]))): ax.scatter(100*x,100*y,100*z, color = 'r')
        print(x)
    plt.show()
'''
GetMap()