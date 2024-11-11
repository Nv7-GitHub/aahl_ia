"""
/*
 * sim.c
 *
 *  Created on: Jun 20, 2024
 *      Author: nv
 */




#include "sim.h"
#include "sensor.h"

// res is in the format az, ax
float calcAccel(float ti, float vzi, float Cd) {
	return -0.5*config.alpha*Cd*pow(vzi, 2)/config.mass - 9.81;
}

const float h = 0.1;

// res is x, vz
void solveIter(float ti, float xi, float vzi, float Cd, float res[2]) {
	float k0 = h*vzi;
	float l0 = calcAccel(ti, vzi, Cd);

	float k1 = h*(vzi+0.5*k0);
	float l1 = calcAccel(ti+0.5*h, vzi+0.5*k0, Cd);

	float k2 = h*(vzi + 0.5*k1);
	float l2 = calcAccel(ti+0.5*h, vzi+0.5*k1, Cd);

	float k3 = h*(vzi+k2);
	float l3 = calcAccel(ti+h, vzi+k2, Cd);

	res[0] = xi + (1.0 / 6.0) * (k0 + 2.0 * k1 + 2.0 * k2 + k3);
	res[1] = vzi + (1.0 / 6.0) * (l0 * h + 2.0 * l1 * h + 2.0 * l2 * h + l3 * h);
}

float getApogee(float ti, float xi, float vzi, float Cd) {
	float vz = vzi;
	float x = xi;
	float t = ti;

	while (t < 0.001f || vz > 0.0f) {
		float res[2];
		solveIter(t, x, vz, Cd, res);
		x = res[0];
		vz = res[1];
		t += h;
	}

	return x;
}
"""

# calcAccel in python:
alpha = 0.003425*1.1682
mass = 0.615
def calcAccel(ti, vzi, Cd):
    return -0.5*alpha*Cd*pow(vzi, 2)/mass - 9.81

# solveIter in python:
def solveIter(ti, xi, vzi, Cd):
    h = 0.1
    k0 = h*vzi
    l0 = calcAccel(ti, vzi, Cd)

    k1 = h*(vzi+0.5*k0)
    l1 = calcAccel(ti+0.5*h, vzi+0.5*k0, Cd)

    k2 = h*(vzi + 0.5*k1)
    l2 = calcAccel(ti+0.5*h, vzi+0.5*k1, Cd)

    k3 = h*(vzi+k2)
    l3 = calcAccel(ti+h, vzi+k2, Cd)

    return xi + (1.0 / 6.0) * (k0 + 2.0 * k1 + 2.0 * k2 + k3), vzi + (1.0 / 6.0) * (l0 * h + 2.0 * l1 * h + 2.0 * l2 * h + l3 * h)

# getApogee in python:
def getApogee(ti, xi, vzi, az):
    vz = vzi
    x = xi
    t = ti
    Cd = abs(-2*mass*(az + 9.81)/(alpha*vz**2));

    while t < 0.001 or vz > 0.0:
        x, vz = solveIter(t, x, vz, Cd)
        t += 0.1

    return x

