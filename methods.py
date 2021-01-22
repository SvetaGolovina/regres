import numpy as np
import pandas as pd
import math

def mnk(x_data, y_data):
    temp1 = np.linalg.inv(mult(transp(xdata), xdata)) # без нампая оч долго
    temp2 = transp(xdata)
    temp3 = mult(temp1, temp2)
    coeffs = mult(temp3, y_data)

    y = []
    for i, data in enumerate(x_data):
        res = mult([data], coeffs)
        y.append(result)
    return coeffs, y

def cal_cost(X, Y, w):
    m=len(Y) 
    predictions = (X * w[:-1]).sum(axis=1) + w[-1] 
    cost = ((Y - predictions) ** 2).sum() 
    return cost 
    
def cal_cost_with_reg(X, Y, w, reg_param, reg_type): 
    m=len(Y) 
    predictions = (X * w[:-1] + w[-1]).sum(axis=1) 
    if reg_type == 'L1': 
        reg_cost=(reg_param * abs(w[:-1])).sum() 
    else: 
        reg_cost=(reg_param * w[:-1]**2).sum() 
    cost =((Y - predictions) ** 2).sum() + reg_cost  
    return cost

def gradient(X,Y, w): 
    h = 2 
    cost_er = cal_cost(X,Y, w) 
    size = len(w) 
    grad = np.zeros(size) 
    for i in range(size): 
        w[i] += h 
        grad[i] = (cost_er - cal_cost(X,Y, w)) 
        w[i] -= h 
    return grad, cost_er 

def gradient_with_reg(X, Y, w, reg_type, reg_param): 
    h = 0.0001 
    cost_er = cal_cost_with_reg(X, Y, w, reg_param, reg_type) 
    size = len(w) 
    grad = np.zeros(size) 
    for i in range(size): 
        w[i] += h 
        grad[i] = (cost_er - cal_cost_with_reg(X, Y, w, reg_param, reg_type)) 
        w[i] -= h 
    return grad, cost_er  

def SGD(X, Y, w, reg_type, reg_param, coef_grad, coef_prev_grad, num_iter=40): 
    if reg_type == 'No': 
        prev_grad, prev_error = gradient(X,Y, w) 
    else: 
        prev_grad, prev_error = gradient_with_reg(X,Y,w, reg_type, reg_param) 
    w = w + prev_grad * coef_grad 
    error = prev_error+1 
    next_error = 0 
    min_err= error 
    opt_w=w 
    for c in range(1, num_iter + 1): 
        if reg_type == "No": 
            grad, error = gradient(X,Y, w) 
            w = w + grad * coef_grad + prev_grad * coef_prev_grad 
            next_error = cal_cost(X, Y, w)  
        else: 
            grad, error = gradient_with_reg(X,Y,w, reg_type, reg_param) 
            w = w + grad * coef_grad + prev_grad * coef_prev_grad 
            next_error = cal_cost_with_reg(X, Y, w, reg_param, reg_type) 
        if next_error > error: 
            coef_grad /= 2 
            coef_prev_grad /= 2 
            w = w - grad * coef_grad + prev_grad * coef_prev_grad 
            if reg_type == "No": 
                next_error = cal_cost(X, Y, w)    
            else: 
                next_error = cal_cost_with_reg(X, Y, w, reg_param, reg_type) 
        else: 
            coef_grad *= 2 
            coef_prev_grad *= 2 
         
        prev_grad = grad 
        prev_error = error 
        if min_err>next_error: 
            min_err=next_error 
            opt_w=w 
        error = next_error 
        #print(f'{c}  :  {error=}') 
        if abs(error - prev_error) <= 10:  
            break 
    return opt_w, cal_cost(X, Y, opt_w) 

def RMSprop(x_data, y_data, coeffs, gamma, num):
    teta = 0.000000001
    g_vec = np.zeros(len(coeffs)) + 0.01
    g_vec_sqr = g_vec ** 2
    next_error = 0

    for k in range(1, num + 1):
        grad, error = gradient(x_data, y_data, coeffs)
        g_vec_sqr = g_vec_sqr * gamma + (1 - gamma) * grad ** 2
        coeffs = coeffs + (grad * teta) / g_vec
        next_error = cal_cost(x_data, y_data, coeffs)

        if next_error > error:
            teta /= 2
            coeffs = coeffs - (grad * teta) / g_vec
        else:
            teta *= 2

        g_vec = g_vec_sqr ** 0.5
        error = next_error
    return coeffs, cal_cost(x_data, y_data, coeffs)

def RMSprop_regul(x_data, y_data, coeffs, reg_type, reg_param, gamma, num):
    teta = 0.000000001
    g_vec = np.zeros(len(coeffs)) + 0.01
    g_vec_sqr = g_vec ** 2
    next_error = 0

    for k in range(1, num + 1):
        grad, error = gradient_with_reg(x_data, y_data, coeffs, reg_type, reg_param)
        g_vec_sqr = g_vec_sqr * gamma + (1 - gamma) * grad ** 2
        coeffs = coeffs + (grad * teta) / g_vec
        next_error = cal_cost_with_reg(x_data, y_data, coeffs, reg_param, reg_type)

        if next_error > error:
            teta /= 2
            coeffs = coeffs - (grad * teta) / g_vec
        else:
            teta *= 2

        g_vec = g_vec_sqr ** 0.5
        error = next_error
    return coeffs, cal_cost(x_data, y_data, coeffs)

def Adagrad(x_data, y_data, coeffs, reg_type, reg_param, num_iter = 50):
    teta = 0.0001
    g_vec = np.zeros(len(coeffs)) + 0.01
    g_vec_sqr = g_vec ** 2
    
    next_error = 0
    for k in range(1, num_iter + 1):
        if reg_type != "No": 
            grad, error = gradient_with_reg(x_data,y_data, coeffs, reg_type, reg_param)
        else:
            grad, error = gradient(x_data,y_data, coeffs)
        g_vec_sqr = g_vec_sqr + grad ** 2
        coeffs = coeffs + (grad * teta) / g_vec
        
        if reg_type != "No": 
            next_error = cal_cost_with_reg(x_data, y_data, coeffs, reg_param, reg_type)
        else:
            next_error = cal_cost(x_data,y_data, coeffs)
        while next_error > error:
            teta /= 2
            coeffs = coeffs - (grad * teta) / g_vec
            if reg_type != "No": 
                next_error = cal_cost_with_reg(x_data, y_data, coeffs, reg_param, reg_type)
            else:
                next_error = cal_cost(x_data,y_data, coeffs)
        else:
            teta *= 2
        g_vec = g_vec_sqr ** 0.5
        error = next_error
    return coeffs, cal_cost(x_data, y_data, coeffs)

def Adam(x_data, y_data, coeffs, reg_type, reg_param, scale_coef_M = 0.5, scale_coef_G = 0.5, num_iter = 50):
    teta = 0.1
    g_vec_sqr = np.zeros(len(coeffs)) + 0.01
    M_vector = np.zeros(len(coeffs))
    next_error = 0

    for k in range(1, num_iter + 1):
        if reg_type != "No": 
            grad, error = gradient_with_reg(x_data, y_data, coeffs, reg_type, reg_param)
        else:
            grad, error = gradient(x_data, y_data, coeffs)
        
        M_vector = M_vector * scale_coef_M + (1 - scale_coef_M) * grad
        g_vec_sqr = g_vec_sqr * scale_coef_G + (1 - scale_coef_G) * grad ** 2
        
        m_ = M_vector / (1 - scale_coef_M ** k)
        g_ = (g_vec_sqr / (1 - scale_coef_G ** k)) ** 0.5
        coeffs = coeffs + m_ * teta / g_
        
        if reg_type != "No": 
            next_error = cal_cost_with_reg(x_data, y_data, coeffs, reg_param, reg_type)
        else:
            next_error = cal_cost(x_data, y_data, coeffs)
        
        while next_error > error:
            teta /= 2
            m_ = M_vector / (1 - scale_coef_M ** k)
            g_ = (g_vec_sqr / (1 - scale_coef_G ** k)) ** 0.5 
            coeffs = coeffs - m_ * teta / g_
      
            if reg_type != "No": 
                next_error = cal_cost_with_reg(x_data, y_data, coeffs, reg_param, reg_type)
            else:
                next_error = cal_cost(x_data, y_data, coeffs)
        else:
            teta *= 2
        error = next_error
    return coeffs, cal_cost(x_data, y_data, coeffs)