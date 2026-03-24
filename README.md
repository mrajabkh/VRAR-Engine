# Tiny 3D Renderer + IMU Bunny Demo

## Overview

This program renders a 3D bunny scene using IMU data to control its orientation.
It runs **two renders in sequence**, each showing:

* a main bunny whose rotation is driven by IMU data
* a falling bunny affected by gravity and collisions
* a floor

The two renders demonstrate different IMU fusion methods.

---

## Required files

All of the following must be in the **same folder**:

* `render.py` — main script
* `bunny.obj` — bunny 3D model
* `floor.obj` — floor 3D model
* `IMUData.csv` — IMU dataset

---

## What the program does

### Render 1: Gyro + Accel

The first window uses:

* gyroscope data
* accelerometer data

This produces a fused orientation for the bunny.

---

### Render 2: Gyro + Accel + Mag

After closing the first window, a second window opens.

This version uses:

* gyroscope
* accelerometer
* magnetometer

This reduces yaw drift and improves overall orientation stability.

---

## Switching between renders

You do not switch manually.

To move to the next render:

* press **Esc**, or
* click the window **X**

Pressing **Esc** will close the current render and automatically open the next one.

After the second render is closed, the program ends.

---

## Camera controls

While a render window is open:

### Rotation

* `A` — rotate camera left
* `D` — rotate camera right
* `W` — rotate camera upward
* `S` — rotate camera downward

### Zoom

* `Q` — zoom in
* `E` — zoom out

### Exit / next render

* `Esc` — close current render and open the next one or exit

