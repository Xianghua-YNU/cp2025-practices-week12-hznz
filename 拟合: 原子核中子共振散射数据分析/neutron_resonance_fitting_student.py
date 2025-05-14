import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def breit_wigner(E, Er, Gamma, fr):
    """
    Breit-Wigner共振公式实现
    
    参数:
        E (float/np.ndarray): 入射能量(MeV)
        Er (float): 共振能量(MeV)
        Gamma (float): 共振宽度(MeV)
        fr (float): 共振强度(mb)
        
    返回:
        float/np.ndarray: 共振截面(mb)
    """
    # 根据公式 f(E) = fr / [(E-Er)^2 + (Gamma^2)/4]
    return fr / ((E - Er)**2 + (Gamma**2) / 4)

def fit_without_errors(energy, cross_section):
    """
    不考虑误差的拟合函数
    
    参数:
        energy (np.ndarray): 能量数组
        cross_section (np.ndarray): 截面数据数组
        
    返回:
        tuple: (最优参数数组, 协方差矩阵)
    """
    # 初始参数猜测 [Er, Gamma, fr]
    initial_guess = [75.0, 50.0, 10000.0]
    
    # 调用curve_fit进行拟合，不传入误差参数
    popt, pcov = curve_fit(
        breit_wigner, 
        energy, 
        cross_section, 
        p0=initial_guess
    )
    return popt, pcov

def fit_with_errors(energy, cross_section, errors):
    """
    考虑误差的拟合函数
    
    参数:
        energy (np.ndarray): 能量数组
        cross_section (np.ndarray): 截面数据数组
        errors (np.ndarray): 误差数组
        
    返回:
        tuple: (最优参数数组, 协方差矩阵)
    """
    # 初始参数猜测 [Er, Gamma, fr]
    initial_guess = [75.0, 50.0, 10000.0]
    
    # 调用curve_fit，传入误差参数并设置absolute_sigma=True
    popt, pcov = curve_fit(
        breit_wigner, 
        energy, 
        cross_section, 
        p0=initial_guess,
        sigma=errors,
        absolute_sigma=True
    )
    return popt, pcov

def plot_fit_results(energy, cross_section, errors, popt, pcov, title):
    """
    绘制拟合结果图表
    
    参数:
        energy (np.ndarray): 能量数据
        cross_section (np.ndarray): 截面数据
        errors (np.ndarray): 误差数据
        popt (np.ndarray): 拟合参数
        pcov (np.ndarray): 协方差矩阵
        title (str): 图表标题
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制实验数据点（带误差棒）
    plt.errorbar(
        energy, cross_section, yerr=errors, 
        fmt='o', color='blue', markersize=5, 
        ecolor='gray', label='Experimental Data'
    )
    
    # 生成拟合曲线数据
    E_fit = np.linspace(energy.min(), energy.max(), 500)
    cross_section_fit = breit_wigner(E_fit, *popt)
    plt.plot(E_fit, cross_section_fit, 'r-', linewidth=2, label='Fitted Curve')
    
    # 计算参数误差（95%置信区间）
    Er, Gamma, fr = popt
    Er_err = np.sqrt(pcov[0, 0]) * 1.96  # 95% CI
    Gamma_err = np.sqrt(pcov[1, 1]) * 1.96
    fr_err = np.sqrt(pcov[2, 2]) * 1.96
    
    # 添加参数标注
    text_str = (
        f'$E_r$ = {Er:.1f} ± {Er_err:.1f} MeV\n'
        f'$\Gamma$ = {Gamma:.1f} ± {Gamma_err:.1f} MeV\n'
        f'$f_r$ = {fr:.0f} ± {fr_err:.0f}'
    )
    plt.text(
        0.05, 0.95, text_str, 
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox={'facecolor': 'white', 'alpha': 0.8}
    )
    
    # 图表修饰
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Cross Section (mb)')
    plt.title(title)
    plt.legend()
    plt.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    return plt.gcf()

def main():
    # 实验数据加载
    energy = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200])
    cross_section = np.array([10.6, 16.0, 45.0, 83.5, 52.8, 19.9, 10.8, 8.25, 4.7])
    errors = np.array([9.34, 17.9, 41.5, 85.5, 51.5, 21.5, 10.8, 6.29, 4.14])
    
    # 任务1：不考虑误差的拟合
    popt1, pcov1 = fit_without_errors(energy, cross_section)
    plot_fit_results(energy, cross_section, errors, popt1, pcov1, 'Breit-Wigner Fit (Without Errors)')
    
    # 任务2：考虑误差的拟合
    popt2, pcov2 = fit_with_errors(energy, cross_section, errors)
    plot_fit_results(energy, cross_section, errors, popt2, pcov2, 'Breit-Wigner Fit (With Errors)')
    
    plt.show()
    
    # 打印结果比较
    print("=== 不考虑误差的拟合结果 ===")
    print(f"共振能量 Er = {popt1[0]:.1f} ± {1.96*np.sqrt(pcov1[0,0]):.1f} MeV")
    print(f"共振宽度 Γ = {popt1[1]:.1f} ± {1.96*np.sqrt(pcov1[1,1]):.1f} MeV")
    print(f"强度参数 fr = {popt1[2]:.0f} ± {1.96*np.sqrt(pcov1[2,2]):.0f}\n")
    
    print("=== 考虑误差的拟合结果 ===")
    print(f"共振能量 Er = {popt2[0]:.1f} ± {1.96*np.sqrt(pcov2[0,0]):.1f} MeV")
    print(f"共振宽度 Γ = {popt2[1]:.1f} ± {1.96*np.sqrt(pcov2[1,1]):.1f} MeV")
    print(f"强度参数 fr = {popt2[2]:.0f} ± {1.96*np.sqrt(pcov2[2,2]):.0f}")

if __name__ == "__main__":
    main()
