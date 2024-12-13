import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from . import ergodic_utils as erg


def plot3D_variety(Xs, ys):
    fig, axs = plt.subplots(1,2,figsize=(10,5))
    axs[0].contour(Xs[:,0], Xs[:,1], ys, cmap="Greys")
    mesh = axs[0].pcolormesh(Xs[:,0], Xs[:,1], ys)
    fig.colorbar(mesh,ax=axs[0])
    mesh2 = axs[1].pcolormesh(Xs[:,0], Xs[:,1], ys)
    fig.colorbar(mesh2,ax=axs[1])

    for i in range(2):
        axs[i].set_xlabel('$x_1$')
        axs[i].set_ylabel('$x_2$')

    fig.show()
    plt.show()

    a = plt.axes(projection='3d')
    a.contour3D(Xs[:,0], Xs[:,1], ys, 50)
    a.set_xlabel('$x_1$')
    a.set_ylabel('$x_2$')
    a.set_zlabel('y')
    a.set_title('3D contour')
    plt.show()

def plotTraj_time(plot, ts, traj, ds, **kwargs):
    (fig, ax) = plot
    ax.plot(ts, traj,kwargs.get("color","black"),linewidth=kwargs.get("linewidth", 2))
    if ds is not None:
        ax.scatter(ts, traj,c=(ds>0)*1,s=30, cmap="coolwarm")
        ax.scatter([0],[2],c="darkred",label="$d>0$")
        ax.scatter([0],[2],c="b",label="$d<0$")
        ax.legend(loc='lower left')

def plotInfo_time(plot, xs, info, tbounds, cbar=False):
    (fig, ax) = plot
    if type(info) is list or type(info) is np.ndarray:
        info = erg.inverse_cks_function_1D(info, 1)

    mu = np.array([info(x) for x in xs])
    (tmin, tmax) = tbounds
    contour_xs = np.vstack((xs, xs))
    contour_zs = np.vstack((mu, mu))
    contour_ys = np.vstack((np.zeros(len(xs)),np.ones(len(xs))*tmax))
    ax.set_xlabel("Time")
    ax.set_ylabel("Position")


    # s = ax.pcolormesh(contour_ys, contour_xs, contour_zs, \
            # vmin=0, vmax=np.max(contour_zs)*1.2,cmap="Reds")
    ax.set_xlim(tbounds)
    if cbar and False:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('top', size='5%', pad=0.05)
        fig.colorbar(s, cax=cax, orientation='horizontal', label="Information Density")
        cax.xaxis.tick_top()
        cax.xaxis.set_label_position('top') 

def plotAll_time(plot, ts, traj, xs, info, cbar=False, ds=None, **kwargs):
    plotInfo_time(plot, xs, info, (np.min(ts), np.max(ts)), cbar)
    plotTraj_time(plot, ts, traj, ds, **kwargs)
    (fig, ax) = plot
    ax.set_ylim(np.min(xs), np.max(xs))

def plotInfo_freq(plot, xs,info, reverse=False):
    (fig, ax) = plot
    if type(info) is list or type(info) is np.ndarray:
        info = erg.inverse_cks_function_1D(info, 1)
    mu = np.array([info(x) for x in xs])
    if reverse:
        ax.plot(xs, mu, 'darkred', linewidth=3, label="Info ($\phi$)")
    else:
        ax.plot(mu, xs, 'darkred', linewidth=3, label="Info ($\phi$)")

def plotTraj_freq(plot, xs, traj, kmax, reverse=False):
    (fig, ax) = plot
    cks = erg.get_cks_1D(traj, 1, kmax)
    traj_decomp = erg.inverse_cks_function_1D(cks, L=1)
    traj_freq = np.array([traj_decomp(x) for x in xs])
    if reverse:
        ax.fill_between(xs, traj_freq, color='grey', label="Time-Avg.\nStats ($c$)")
    else:
        ax.fill_betweenx(xs, traj_freq, color='grey', label="Time-Avg\nStats ($c$)")
        

def plotAll_freq(plot, xs, info, traj, kmax, cbar=False, reverse=False):
    plotInfo_freq(plot, xs, info, reverse=reverse)
    plotTraj_freq(plot, xs, traj, kmax, reverse=reverse)
    (fig, ax) = plot
    if reverse:
        # ax.set_ylim(bottom=0)
        # # ax.set_xlim(left=0.1)
        # ax.set_xlim(np.min(xs), np.max(xs))
        ax.set_ylabel("Probability\nDensity")
        # ax.set_xlabel("Position")
        ax.set_xlim(np.min(xs), np.max(xs))
        ax.set_ylim(bottom=0)
        ax.tick_params(labelbottom=False, bottom=True)
        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cax.axis("off")
            return cax
    else:
        ax.set_xlim(left=0)
        # ax.set_xlim(left=0.1)
        ax.set_ylim(np.min(xs), np.max(xs))
        ax.set_xlabel("Probability\nDensity")
        ax.set_ylabel("Position")
        ax.yaxis.set_label_position('right') 
        ax.tick_params(labelright=True, right=True)
        if cbar and False:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('top', size='5%', pad=0.05)
            cax.axis("off")


def plotAll_both(plot, ts, traj, xs, info, kmax, cbar, ds=None, **kwargs):
    (fig, axes) = plot
    plotAll_freq((fig, axes[1]), xs, info, traj, kmax, cbar)
    plotAll_time((fig, axes[0]),ts, traj, xs, info, cbar, ds, **kwargs)
    plt.subplots_adjust(wspace=0.06)            

def plot_ext(plot, ts, input):
    (fig, ax) = plot
    ax.plot(ts, input)
    ax.set_xlim((np.min(ts), np.max(ts)))

def setup_plot(input_num):
    vcount = 1 + input_num
    # fig = plt.figure(figsize=(4+1.2*input_num, 6))
    fig = plt.figure(figsize=(4+1.2*input_num, 5))
    # fig = plt.figure(figsize=(3+1.7*input_num, 4))
    height_ratios = [1] * vcount
    height_ratios[0] = 4
    gs = fig.add_gridspec(vcount,2, wspace=0.05, hspace=0.075, width_ratios=[3,1], height_ratios=height_ratios)
    axs = gs.subplots()
    for i in range(1,input_num+1):
        axs[i,1].axis("off")
    return (fig, axs)

def plotAll_both_ud(plot, ts, traj, xs, info, us, ds, kmax, cbar):
    (fig, axes) = plot
    print(axes)
    if len(axes.ravel()) == 2:  # Plotting on single axes
        kwargs = {"color": "gray", "linewidth": 1}
    else:
        kwargs = {}

    plotAll_both((fig, axes[0]), ts, traj, xs, info, kmax, cbar, ds, **kwargs)
    if len(axes.ravel()) == 4:  # Plotting ctrl on separate axes
        plot_ext((fig, axes[1,0]), ts, us)
        axes[1,0].set_ylabel("Control")
    axes[0,0].get_xaxis().set_ticklabels([])
    axes[0,1].get_yaxis().set_ticklabels([])
    axes[0,1].tick_params(right=False)
    axes[0,1].set_ylabel(None)

    # if ds is not None:

    #     plot_ext((fig, axes[2,0]), ts, ds)
    #     axes[1,0].get_xaxis().set_ticklabels([])
    #     # axes[0,1].get_yaxis().set_ticklabels([])
    #     axes[2,0].set_ylabel("Disturbance")
    #     axes[2,0].set_xlabel("Time")
    # else:
    axes[0,0].set_xlabel("Time")


def evaluate_traj_and_plot(ts, traj, xs, info, us, ds, kmax, cbar, plot=None):
    if plot is None:
        if ds is None:
            plot = setup_plot(1)
        else:
            plot = setup_plot(2)

        fig = plt.figure(figsize=(6, 4))
        gs = fig.add_gridspec(2,2, wspace=0.1, hspace=0.1, width_ratios=[3,1], height_ratios=[1,0.3])
        axs = gs.subplots()
        # axs[1,0].axis("off")
        axs[1,1].axis("off")
        plot =  (fig, axs)
    else:
        fig, axs = plot
    
    plotAll_both_ud(plot, ts, traj, xs, info, us, ds, kmax, cbar)
    axs[0,1].set_yticklabels([])
    # axs[0,0].set_title("Exploration (Uniform Info.)")
    # axs[0,1].set_title("Time-Averaged\nStatistics")
    return fig, axs


#@############################################################################################################################################
#@############################################################################################################################################
#@############################################################################################################################################
#@############################################################################################################################################
#@############################################################################################################################################


def plot_with_recon(ts, traj, xs, traj_recon, mu_recon=None,emetr=None,kmax=None, axes=None):
    fig = plt.figure()
    gs = fig.add_gridspec(1,2, wspace=0.0, width_ratios=[3,1])#, left=0.1, right=0.9, top=0.9, bottom=0.1)
    if axes==None:
        axs = gs.subplots()
    else:
        axs = axes
    # axs = [0, 0]
    # axs[0] = fig.add_subplot(gs[0])
    # axs[1] = fig.add_subplot(gs[1], sharey=axs[0])

    axs[1].get_shared_y_axes().join(axs[0], axs[1])
    axs[1].set_yticklabels([])
    # if emetr is not None:
    #     fig.suptitle(r"Trajectory and Time-Average Statistics ($n={:d},\scrE={:.3f}$)".format(kmax,emetr))
    # else:
    #     fig.suptitle("Trajectory and Time-Average Statistics")
    axs[0].plot(ts, traj,'k',linewidth=3)
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Position ($m$)")
    
    contour_xs = np.vstack((xs, xs))
    contour_zs = np.vstack((mu_recon, mu_recon))
    contour_ys = np.vstack((np.zeros(len(xs)),np.ones(len(xs))*np.max(ts)))
    axs[0].pcolormesh(contour_ys, contour_xs, contour_zs, \
            vmin=-np.max(contour_zs)/2, vmax=np.max(contour_zs)*1.5,cmap="Greys")
    axs[0].set_xlim([0,np.max(ts)])
    # axs[1].set_ylim([0,1])
    axs[1].plot(traj_recon, xs,'r',label="Actual\n(Approximation)")
    if mu_recon is not None:
        axs[1].plot(mu_recon, xs, 'k',linewidth=2,label="Desired")
    axs[1].set_xlabel("Time-Averaged\nPDF")
    axs[1].legend()
    axs[1].fill_between(np.append([0],np.append(traj_recon,[0])),
                        np.append([0],np.append(xs,[1])),color='r')
    axs[1].set_xlim(left=0)

    # print("Traj sum: ",np.average(traj_recon))
    # if mu_recon is not None:
    #     print(np.sum("Mu sum: ", np.average(mu_recon)))
    plt.tight_layout()
    plt.show()

def plot_with_recon_ud(ts, traj, xs, traj_recon, mu_recon=None,emetr=None,kmax=None, us=None, ds=None, axes=None):
    vcount = 1 + (us is not None) + (ds is not None)
    if np.array([axes==None]).all():
        fig = plt.figure()
        height_ratios = [1] * vcount
        height_ratios[0] = 4
        gs = fig.add_gridspec(vcount,2, wspace=0.05, hspace=0.075, width_ratios=[3,1], height_ratios=height_ratios)#, left=0.1, right=0.9, top=0.9, bottom=0.1)
        axs = gs.subplots()
    else:
        axs = axes
    if vcount == 1:
        axs = [axs,0]
    axs[0][1].get_shared_y_axes().join(axs[0][0], axs[0][1])
    axs[0][1].set_yticklabels([])
    axs[0][1].yaxis.tick_right()
    axs[0][1].tick_params(axis="y",direction="in", pad=-22,length=5)
    if vcount > 1:
        axs[1,0].get_shared_x_axes().join(axs[0,0], axs[1,0])
        axs[0,0].set_xticklabels([])
        axs[0][0].xaxis.tick_top()
        axs[0][0].tick_params(axis="x",direction="in", pad=-15,length=5)

        # axs[0,0].set_xticklabels([]
        axs[1,1].axis('off')
        if vcount == 3:
            axs[2,0].get_shared_x_axes().join(axs[0,0], axs[2,0])
            axs[1,0].set_xticklabels([])
            axs[2,0].set_xlabel("Time ($s$)")
            axs[2,1].axis('off')

        else:
            axs[1,0].set_xlabel("Time ($s$)")
    else:
        axs[0][0].set_xlabel("Time ($s$)")

    plt.rcParams['axes.titley'] = 1.0    # y is in axes-relative co-ordinates.
    # plt.rcParams['axes.titlex'] = 0.0    # y is in axes-relative co-ordinates.

    plt.rcParams['axes.titlepad'] = -14  # pad is in points...
    axs[0,0].set_title(" a)", loc='left')
    axs[0,1].set_title(" b)", loc='left')
    axs[1,0].set_title(" c)", loc='left')

    # if emetr is not None:
    #     # fig.suptitle(r"Trajectory and Time-Average Statistics ($n={:d},\scrE={:.3f}$)".format(kmax,emetr))
    #     axs[0,0].set_title('Trajectory and Time-Average Statistics\n($n={:d},\scrE={:.3f}$)'.format(kmax,emetr))
    # else:
    #     axs[0,0].set_title("Trajectory and Time-Average Statistics")
    #     # fig.suptitle("Trajectory and Time-Average Statistics")

    axs[0][0].plot([], [],'w',linewidth=3, label= r'$a)$')
    axs[0][1].plot([], [],'w',linewidth=3, label= r'$b)$')
    axs[1][0].plot([], [],'w',linewidth=3, label= r'$c)$')
    axs[0][0].plot(ts, traj,'k',linewidth=3, label="Trajectory")
    axs[0][0].set_ylabel("Position ($m$)")
    
    contour_xs = np.vstack((xs, xs))
    contour_zs = np.vstack((mu_recon, mu_recon))
    contour_ys = np.vstack((np.zeros(len(xs)),np.ones(len(xs))*np.max(ts)))
    axs[0][0].pcolormesh(contour_ys, contour_xs, contour_zs, \
            vmin=0, vmax=np.max(contour_zs)*1.5,cmap="Reds", label="Information Density")
    axs[0][0].fill_between([],[],color='orangered', label="Information Density")
            # vmin=-np.max(contour_zs)/2, vmax=np.max(contour_zs)*2.5,cmap="Reds", label="Information Density")
    axs[0][0].set_xlim([0,np.max(ts)+ts[1]-ts[0]])
    print(np.max(ts))
    axs[0][0].set_ylim([0, 1])
    # axs[1].set_ylim([0,1])
    axs[0][1].plot(traj_recon, xs,color='darkgray')
    axs[0][1].set_xlabel("Relative Amount of\nSearch Time or Information")
    axs[0][1].set_xlim([0, 1.75])
    # axs[0][1].legend(loc='best',bbox_to_anchor=(0.5,0,0.5,-0.2))
    axs[0][1].fill_between(np.append([0],np.append(traj_recon,[0])),
                        np.append([0],np.append(xs,[1])),color='darkgray', label="Search time")
    axs[0][1].set_xlim(left=0)
    if mu_recon is not None:
        axs[0][1].plot(mu_recon, xs, color='tab:red',linewidth=3,label="Information Density")

    if us is not None:
        axs[1,0].plot(ts, us, label="Control Signal")
        axs[1,0].set_ylabel("Control Signal\n(Acceleration, $m/s^2$)")
        if ds is not None:
            axs[2,0].plot(ts, ds, label="Disturbance")
            axs[2,0].set_ylabel("$d$") 
    elif ds is not None:
        axs[1,0].plot(ts, ds, "Disturbance")
        axs[1,0].set_ylabel("$d$")

    # print("Traj sum: ",np.average(traj_recon))
    # if mu_recon is not None:
    #     print(np.sum("Mu sum: ", np.average(mu_recon)))
    # plt.tight_layout()
    plt.show()

def random_double_integrator(steps, min, max):
    a = np.random.random(steps) * 2 - 1
    v, s = a*0, a*0
    v[1] = a[0]
    for i in range(2, len(a)):
        v[i] = a[i-1] + v[i-1]*0.75
        s[i] = v[i-1] + s[i-1]

    srange = np.max(s)-np.min(s) / (max-min)
    return (s-np.min(s)+min)/srange, v/srange, a/srange

def evaluate_traj_and_show(t,s,kmax,xs=np.linspace(0,1,101),mus=None, us=None, ds=None, axes=None):
    cks = erg.get_cks_1D(s, 1, kmax)
    traj_recon = erg.inverse_cks_1D(cks, xs, 1)
    if mus is not None:
        phiks = erg.get_phiks_1D(xs,mus,1,kmax)
        # print("plotter phiks:", phiks)
        mu_recon = erg.inverse_cks_1D(phiks, xs, 1)
        Erg_metr=0
        for k in range(len(cks)):
            # print("c_{:d} = {:+.04f}\tphi_{:d} = {:+.04f}\tDiff = {:+.04f}".format(k+1,cks[k],k+1,phiks[k],cks[k]-phiks[k]))
            # print("Diff {:d} = {:+.04f}".format(k,cks[k]-phiks[k]), end="\t")
            term_to_add = (cks[k]-phiks[k])**2 / (1+(k))
            # print("Term to add: {:.04f}".format(term_to_add))
            # print("(c,phi)_{}\t= {:+.03f},\t{:+03f}\tdiff={:+.03f}".format(k,cks[k],phiks[k],cks[k]-phiks[k]))
            Erg_metr += term_to_add
        # print("Ck-based Metric: {:.05f}".format(Erg_metr))
        # Erg_metr /= t[-1]**2
        # print("Ergodic metric: {}".format(Erg_metr))
        # plot_with_recon(t,s,xs,traj_recon,mu_recon,Erg_metr,kmax)
        plot_with_recon_ud(ts=t, traj=s, xs=xs, traj_recon=traj_recon, mu_recon=mu_recon,emetr=Erg_metr,kmax=5, us=us, ds=ds, axes=axes)
    else:
        for k in range(len(cks)):
            e = '\n' if k%3==2 else '\t'
            # print("c_{} = {:.03f}".format(k,cks[k]),end=e)
        plot_with_recon_ud(ts=t,traj=s,xs=xs,traj_recon=traj_recon, us=us, ds=ds, axes=axes)
    return Erg_metr
    # print(cks[1:])
    # print(cks[1:] / cks[1])
    

if __name__ == "__main__":
    # evaluate a random trajectory 
    s, v, a = random_double_integrator(101,0,1)
    xs = np.linspace(0,1,101)
    ts = np.arange(101)
    mu = np.ones(len(xs))
    evaluate_traj_and_show(ts,s,15,xs=xs,mus=mu, us=ts*0)
