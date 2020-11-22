import numpy as np
# np.cross za vektorski proizvod, potrebni nizovi

def homogene(P):
    return [P[0], P[1], 1]

def afine(P):
    return [round(P[0] / P[2]), round(P[1] / P[2])]

def nevidljivo(P1, P2, P3, P5, P6, P7, P8):
    # Prebacivanje afinih koordinata u homogene
    P1 = homogene(P1)
    P2 = homogene(P2)
    P3 = homogene(P3)
    P5 = homogene(P5)
    P6 = homogene(P6)
    P7 = homogene(P7)
    P8 = homogene(P8)

    # Trazi se presek ivica P7P6 i P3P2, presek je tacka P
    P7P6 = np.cross(P7, P6)
    P3P2 = np.cross(P3, P2)

    P = np.cross(P7P6, P3P2)
    
    # Trazi se presek ivica P6P5 i P7P8, presek je tacka Q
    P6P5 = np.cross(P6, P5)
    P7P8 = np.cross(P7, P8)

    Q = np.cross(P6P5, P7P8)

    # Trazenu tacku P4 dobijamo kao presek ivica
    # PP1 i P3Q
    PP1 = np.cross(P, P1)
    P3Q = np.cross(P3, Q)

    P4 = np.cross(PP1, P3Q)
        
    # Vracanje homogenih koordinata u afine
    P4 = afine(P4)
    
    return P4

if __name__ == '__main__':
    # Tacke sa moje slike kutije
    P1 = [697, 371]
    P2 = [502, 577]
    P3 = [68, 280]
    P5 = [740, 260]
    P6 = [521, 477]
    P7 = [17, 163]
    P8 = [251, 18]
    
    P4 = nevidljivo(P1, P2, P3, P5, P6, P7, P8)

    print(f'Koordinate nevidljive tacke sa moje slike ({P4[0]}, {P4[1]})')

    # Tacke sa slike sa sajta
    P1 = [595, 301]
    P2 = [292, 517]
    P3 = [157, 378]
    P5 = [666, 116]
    P6 = [304, 295]
    P7 = [135, 163]
    P8 = [509, 43]
    
    P4 = nevidljivo(P1, P2, P3, P5, P6, P7, P8)

    print(f'Koordinate nevidljive tacke sa slike sa sajta: ({P4[0]}, {P4[1]})')
