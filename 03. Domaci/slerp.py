import numpy as np
from numpy import linalg as LA
import math
import matplotlib.pyplot as plt
from matplotlib import animation
import mpl_toolkits.mplot3d.axes3d as Axes3D

def Euler2A(phi, theta, psi):
    # A = Rz(psi) * Ry(theta) * Rx(phi)
    
    Rz = np.array([
        [math.cos(psi), -math.sin(psi), 0],
        [math.sin(psi), math.cos(psi), 0],
        [0, 0, 1]
    ])
    
    Ry = np.array([
        [math.cos(theta), 0, math.sin(theta)],
        [0, 1, 0],
        [-math.sin(theta), 0, math.cos(theta)]
    ])
    
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(phi), -math.sin(phi)],
        [0, math.sin(phi), math.cos(phi)]
    ])
    
    return Rz.dot(Ry).dot(Rx)

# Dobija vektor kao ulaz a vraca jedinicni vektor u
def normalizacija(u):
    norma = 0
    for x in u:
        norma += x ** 2
    
    norma = math.sqrt(norma)
    
    return u / norma

def AxisAngle(A):
    if round(LA.det(A)) != 1:
        print("Determinanta je razlicita od 1")
        return

    if np.any(np.round(A.dot(A.T),6) != np.eye(3)):
        print("Matrica A nije ortogonalna")
        return
    
    A_E = A - np.eye(3)
    first = A_E[0]
    second = A_E[1]
    third = A_E[2]
    
    p = np.cross(first, second)
    if not np.any(p):
        p = np.cross(first, third)
        if not np.any(p):
            p = np.cross(second, third)
            
    p = normalizacija(p)
    
    # Vektor u je normalan na vektor p
    u = first
    if not np.any(u):
        u = second
        if not np.any(u):
            u = third
            
    u = normalizacija(first)
    
    u_p = A.dot(u)
    u_p = normalizacija(u_p)
    
    # Vektori u i u_p su jedinci pa ne mora da se deli sa proizvodom njihovih normi
    phi = math.acos(u.dot(u_p))
    
    mesoviti_proizvod = LA.det(np.array([u, u_p, p]))
    
    if mesoviti_proizvod < 0:
        p = -p
        
    return (p, phi)

def AxisAngle2Q(p, phi):
    w = math.cos(phi / 2)
    
    p = normalizacija(p)
    
    [x, y, z] = math.sin(phi / 2) * p
    
    return np.array([x, y, z, w])

def lerp(q1, q2, tm, t):
    q = (1 - (t / tm)) * q1 + (t / tm) * q2
    return q

# Vraca jedinicni kvaternion koji zadaje orijentaciju u trenutku t
def slerp(q1, q2, tm, t):
    cos0 = np.dot(q1, q2)

    if cos0 < 0:
        q1 = -q1
        cos0 = -cos0
        
    if cos0 > 0.95:
        return lerp(q1, q2, tm, t)
    
    phi0 = math.acos(cos0)
    
    a = math.sin(phi0 * (1 - t / tm)) / math.sin(phi0)
    b = math.sin(phi0 * t / tm) / math.sin(phi0)

    q_t = a * q1 + b * q2
    return q_t

# Inverz kvaterniona
def qinv(q):
    return [-q[0], -q[1], -q[2], q[3]] / (LA.norm(q) ** 2)
    
# Mnozenje kvaterniona
def qmul(q1, q2):
    v1 = q1[0:3]
    w1 = q1[3]
    v2 = q2[0:3]
    w2 = q2[3]
    v = np.cross(v1, v2) + w2 * v1 + w1 * v2
    w = w1 * w2 - np.dot(v1, v2)
    return np.array([v[0], v[1], v[2], w])

# Rotacija kvaterniona preko q * p * q^-1. p mora da bude kvaternion pa zato dodajemo 0 na kraj (w=0)
def transform(p, q):
    p = np.array([p[0], p[1], p[2], 0.0])
    return qmul(qmul(q, p), qinv(q))[:-1]
    
if __name__ == "__main__":
    tm = 100
    
    # Pocetna pozicija
    p_s = np.array([7, 5, 6])

    # Rotacija
    A_s = Euler2A(math.pi / 3, math.pi / 2, -math.pi / 4)
    # Pocetni kvaternion
    u, angle = AxisAngle(A_s)
    q_s = AxisAngle2Q(u, angle)
    
    # Krajnja pozicija
    p_e = np.array([2, 1, 4])

    # Rotacija
    A_e = Euler2A(-math.pi / 3, math.pi / 6, -math.pi / 9)
    # Krajnji kvaternion
    u, angle = AxisAngle(A_e)
    q_e = AxisAngle2Q(u, angle) 
    
    # Namestanje da bude 3D
    fig = plt.figure(figsize = (7, 7))
    ax = Axes3D.Axes3D(fig)
    
    # Postavljanje granica osa
    ax.set_xlim3d([0.0, 10.0])
    ax.set_xlabel('X osa')

    ax.set_ylim3d([0.0, 10.0])
    ax.set_ylabel('Y osa')

    ax.set_zlim3d([0.0, 10.0])
    ax.set_zlabel('Z osa')

    ax.view_init(10, -5)

    colors = ['r', 'g', 'b']

    axis = np.array(sum([ax.plot([], [], [], c=c) for c in colors], []))

    # Ovo su pocetne i krajnje tacke duzi od kojih krecemo
    startpoints = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    endpoints = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    
    # Iscrtavanje pocetne i krajnje pozicije
    # Inicijalno sve u koord. pocetku, pa kada se primeni transformacija treba da se translira
    for i in range(3):
        start = transform(startpoints[i], q_s)
        end = transform(endpoints[i], q_s)
        start += p_s
        end += p_s
	    
        # Iscrtavamo duz na grafiku
        ax.plot([start[0], end[0]], [start[1], end[1]], zs=[start[2], end[2]], color=colors[i])

        start = transform(startpoints[i], q_e)
        end = transform(endpoints[i], q_e)
        start += p_e
        end += p_e
        ax.plot([start[0], end[0]], [start[1], end[1]], zs=[start[2], end[2]], color=colors[i])

    # Init funkcija za animaciju
    def init():
        for a in axis:
            a.set_data(np.array([]), np.array([]))
            a.set_3d_properties(np.array([]))

        return axis

    def animate(frame):
        q = slerp(np.array(q_s), np.array(q_e), tm, frame)
	    
        # Korak koji se dodaje tackama vektora da bi se konstantno translirale ka krajnjim tackama a ne samo rotirale u koord. pocetku
        korak = frame * (p_e - p_s) / tm
	    
        for a, start, end in zip(axis, startpoints, endpoints):
            start = transform(np.array(start), np.array(q))
            end = transform(np.array(end), np.array(q))
            start += p_s + korak
            end += p_s + korak

            a.set_data(np.array([start[0], end[0]]), np.array([start[1], end[1]]))
            a.set_3d_properties(np.array([start[2], end[2]]))
        
        fig.canvas.draw()
        return axis

    anim = animation.FuncAnimation(fig, animate, frames=tm, init_func=init, interval=5, repeat=True, repeat_delay=20)
    anim.save('animation.gif')
    #plt.show()
