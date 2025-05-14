import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def load_supernova_data(file_path):
    """
    从文件中加载超新星数据
    """
    data = np.loadtxt(file_path, comments='#')
    z = data[:, 0]
    mu = data[:, 1]
    mu_err = data[:, 2]
    return z, mu, mu_err

def hubble_model(z, H0):
    """
    哈勃模型：距离模数与红移的关系
    """
    c = 299792.458  # 光速，单位：km/s
    return 5 * np.log10(c * z / H0) + 25

def hubble_model_with_deceleration(z, H0, a1):
    """
    包含减速参数的哈勃模型
    """
    c = 299792.458  # 光速，单位：km/s
    term = 1 + 0.5 * (1 - a1) * z
    return 5 * np.log10(c * z / H0 * term) + 25

def hubble_fit(z, mu, mu_err):
    """
    使用最小二乘法拟合哈勃常数
    """
    def model_func(z, H0):
        return hubble_model(z, H0)
    popt, pcov = curve_fit(model_func, z, mu, sigma=mu_err, absolute_sigma=True, p0=[70])
    H0 = popt[0]
    H0_err = np.sqrt(pcov[0][0])
    return H0, H0_err

def hubble_fit_with_deceleration(z, mu, mu_err):
    """
    使用最小二乘法拟合哈勃常数和减速参数
    """
    def model_func(z, H0, a1):
        return hubble_model_with_deceleration(z, H0, a1)
    popt, pcov = curve_fit(model_func, z, mu, sigma=mu_err, absolute_sigma=True, p0=[70, 1])
    H0, a1 = popt
    H0_err = np.sqrt(pcov[0][0])
    a1_err = np.sqrt(pcov[1][1])
    return H0, H0_err, a1, a1_err

def plot_hubble_diagram(z, mu, mu_err, H0):
    """
    绘制哈勃图（距离模数vs红移）
    """
    plt.figure()
    z_model = np.linspace(z.min(), z.max(), 100)
    mu_model = hubble_model(z_model, H0)
    plt.errorbar(z, mu, yerr=mu_err, fmt='o', label='Supernova data')
    plt.plot(z_model, mu_model, 'r-', label=f'Best fit: $H_0$ = {H0:.2f} km/s/Mpc')
    plt.xlabel('Redshift z')
    plt.ylabel('Distance modulus μ')
    plt.legend()
    plt.grid(True)
    return plt.gcf()

def plot_hubble_diagram_with_deceleration(z, mu, mu_err, H0, a1):
    """
    绘制包含减速参数的哈勃图
    """
    plt.figure()
    z_model = np.linspace(z.min(), z.max(), 100)
    mu_model = hubble_model_with_deceleration(z_model, H0, a1)
    plt.errorbar(z, mu, yerr=mu_err, fmt='o', label='Supernova data')
    plt.plot(z_model, mu_model, 'r-', label=f'Best fit: $H_0$ = {H0:.2f}, $a_1$ = {a1:.2f}')
    plt.xlabel('Redshift z')
    plt.ylabel('Distance modulus μ')
    plt.legend()
    plt.grid(True)
    return plt.gcf()

if __name__ == "__main__":
    # 数据文件路径
    data_file = "E:\load dawm\supernova_data.txt"
    
    # 加载数据
    z, mu, mu_err = load_supernova_data(data_file)
    
    # 拟合哈勃常数
    H0, H0_err = hubble_fit(z, mu, mu_err)
    print(f"拟合得到的哈勃常数: H0 = {H0:.2f} ± {H0_err:.2f} km/s/Mpc")
    
    # 绘制哈勃图
    fig = plot_hubble_diagram(z, mu, mu_err, H0)
    plt.show()
    
    # 可选：拟合包含减速参数的模型
    H0, H0_err, a1, a1_err = hubble_fit_with_deceleration(z, mu, mu_err)
    print(f"拟合得到的哈勃常数: H0 = {H0:.2f} ± {H0_err:.2f} km/s/Mpc")
    print(f"拟合得到的a1参数: a1 = {a1:.2f} ± {a1_err:.2f}")
    
    # 绘制包含减速参数的哈勃图
    fig = plot_hubble_diagram_with_deceleration(z, mu, mu_err, H0, a1)
    plt.show()
