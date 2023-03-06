import matplotlib.pyplot as plt
import numpy as np
import pickle
import sympy
from argparse import ArgumentError
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from utils import functions, pretty_print, schedules

ACTIVATION_FUNCS = [
    *[functions.Constant()] * 2,
    *[functions.Identity()] * 4,
    *[functions.Square()] * 4,
    *[functions.Sin()] * 2,
    # *[functions.Exp()] * 2,
    # *[functions.Sigmoid()] * 2,
    *[functions.Product()] * 2
]


"""
Functions to extract coefficients form sympy expressions.
"""
def id_parser(sympy_expr):
    """Extract coefficients of sympy expressions of form a(t)*x"""
    return [sympy_expr.coeff('x')]

def f1_parser(sympy_expr):
    """Extract coefficients of sympy expressions of form a(t)*x^2+b(t)*x"""
    return [sympy_expr.coeff('x**2'), sympy_expr.coeff('x')]

def sinfreq_parser(sympy_expr):
    """Extract coefficients of sympy expressions of form sin(a(t)*x)"""
    return [sympy.series(sympy_expr, sympy.Symbol('x'), 0, 2).coeff('x')]

def addiff_parser(sympy_expr):
    """Extract coefficients of sympy expressions of the advection-diffusion equation"""
    return [sympy_expr.coeff('x'), sympy_expr.coeff('y'), sympy_expr.coeff('z')]

def burger_parser(sympy_expr):
    """Extract coefficients of sympy expressions of Burgers' equation"""
    return [sympy_expr.coeff('x*y'), sympy_expr.coeff('z')]

def ks_parser(sympy_expr):
    """Extract coefficients of sympy expressions of the Kuramoto-Sivashinsky equation"""
    return [sympy_expr.coeff('x*y'), sympy_expr.coeff('z'), sympy_expr.coeff('a')]

def force_parser(sympy_expr):
    """Extract coefficients of sympy expressions of form a(t)*x-b(t)*y"""
    return [sympy_expr.coeff('x'), -sympy_expr.coeff('y')]

def potential_parser(sympy_expr):
    """Extract coefficients of sympy expressions of form a(t)*x^2+b(t)*y^2-2c(t)*xy"""
    return [sympy_expr.coeff('x**2'), sympy_expr.coeff('y**2'), -sympy_expr.coeff('x*y')/2]

def t_fun(t):
    return np.piecewise(t, [t < 0, (0 <= t) * (t < 1.5), t >= 1.5],
                        [lambda t: 1/2*(5+t), lambda t: 2.5 - 0.5 * t, lambda t: t - 1.5 + 1.75])


"""
Hardcoded information for the plots. Dictionary keys are the function names as found in the results directory.
'output_label': function label in the filename of the output plot
'parser': the appropriate coefficient-extracting function to use for the function
'coeff_names': list of coefficient names, can be anything; ith entry in coeff_names must correspond to ith entry in the list returned by parser
'true_coeff_funcs': dictionary with keys from coeff_names and values equal to lambda functions of the true coefficient as a function of t or x
'coeff_labels': dictionary with keys from coeff_names and values equal to the tuple (label for true coeff function on plot, label for predicted coeff function on plot)
"""
FUNC_INFO_DICTS = {
    't*x': {
        'output_label': 'contr-id',
        'parser': id_parser,
        'coeff_names': ['a(t)'],
        'true_coeff_funcs': {'a(t)': lambda t: t},
        'coeff_labels': {'a(t)': (r'$a(t)$', r'$\hat{a}(t)$')},
        'xlabel': 't'
    },
    't*x**2+3*sin(t)*x': {
        'output_label': 'contr-f1sin',
        'parser': f1_parser,
        'coeff_names': ['a(t)', 'b(t)'],
        'true_coeff_funcs': {'a(t)': lambda t: t, 'b(t)': lambda t: 3*np.sin(t)},
        'coeff_labels': {'a(t)': (r'$a(t)$', r'$\hat{a}(t)$'), 'b(t)': (r'$b(t)$', r'$\hat{b}(t)$')},
        'xlabel': 't'
    },
    't*x**2+3*sgn(t)*x': {
        'output_label': 'contr-f1sgn',
        'parser': f1_parser,
        'coeff_names': ['a(t)', 'b(t)'],
        'true_coeff_funcs': {'a(t)': lambda t: t, 'b(t)': lambda t: 3*np.sign(t)},
        'coeff_labels': {'a(t)': (r'$a(t)$', r'$\hat{a}(t)$'), 'b(t)': (r'$b(t)$', r'$\hat{b}(t)$')},
        'xlabel': 't'      
    },
    'sin(0.5*(5+t)*x)': {
        'output_label': 'contr-sinfreq',
        'parser': sinfreq_parser,
        'coeff_names': ['a(t)'],
        'true_coeff_funcs': {'a(t)': lambda t: 1/2*(5+t)},
        'coeff_labels': {'a(t)': (r'$a(t)$', r'$\hat{a}(t)$')},
        'xlabel': 't'      
    },
    'spring_force': {
        'output_label': 'conv-force',
        'parser': force_parser,
        'coeff_names': ['a(t)', 'b(t)'],
        'true_coeff_funcs': {'a(t)': lambda t: 1/2*(5-t), 'b(t)': lambda t: 1/2*(5-t)},
        'coeff_labels': {'a(t)': (r'$k(t)$', r'$\hat{k}_1(t)$'), 'b(t)': (r'$k(t)$', r'$\hat{k}_2(t)$')},
        'xlabel': 't'
    },
    'spring_potential': {
        'output_label': 'conv-potential',
        'parser': potential_parser,
        'coeff_names': ['a(t)', 'b(t)', 'c(t)'],
        'true_coeff_funcs': {'a(t)': lambda t: 1/2*(5-t), 'b(t)': lambda t: 1/2*(5-t), 'c(t)': lambda t: 1/2*(5-t)},
        'coeff_labels': {'a(t)': (r'$k(t)$', r'$\hat{k}_1(t)$'), 'b(t)': (r'$k(t)$', r'$\hat{k}_2(t)$'), 'c(t)': (r'$k(t)$', r'$\hat{k}_3(t)$')},
        'xlabel': 't'      
    },
    'addiff': {
        'output_label': 'diffeq-addiff',
        'parser': addiff_parser,
        'coeff_names': ['a(x)', 'b(x)', 'c(x)'],
        'true_coeff_funcs': {'a(x)': lambda x: -2*np.pi/5*np.sin(2*np.pi*x/5), 'b(x)': lambda x: -1.5+np.cos(2*np.pi*x/5), 'c(x)': lambda x: 0.1*np.ones_like(x)},
        'coeff_labels': {'a(x)': (r"$f'(x)$", r"$\hat{f}'(x)$"), 'b(x)': (r"$f(x)$", r"$\hat{f}(x)$"), 'c(x)': (r"$\epsilon(x)$", r"$\hat{\epsilon}(x)$")},
        'xlabel': 'x'
    },
    'burgers': {
        'output_label': 'diffeq-burgers',
        'parser': burger_parser,
        'coeff_names': ['a(t)', 'b(t)'],
        'true_coeff_funcs': {'a(t)': lambda t: -(1+np.sin(t)/4), 'b(t)': lambda t: 0.1*np.ones_like(t)},
        'coeff_labels': {'a(t)': (r"$f(t)$", r"$\hat{f}(t)$"), 'b(t)': (r"$\epsilon(t)$", r"$\hat{\epsilon}(t)$")},
        'xlabel': 't'
    },
    'ks': {
        'output_label': 'diffeq-ks',
        'parser': ks_parser,
        'coeff_names': ['a(x)', 'b(x)', 'c(x)'],
        'true_coeff_funcs': {'a(x)': lambda x: 1+np.sin(2*np.pi*x/10)/4, 'b(x)': lambda x: -1+np.exp(-((x-2)**2)/5)/4, 'c(x)': lambda x: -1-np.exp(-((x-2)**2)/5)/4},
        'coeff_labels': {'a(x)': (r"$f(x)$", r"$\hat{f}(x)$"), 'b(x)': (r"$g(x)$", r"$\hat{g}(x)$"), 'c(x)': (r"$h(x)$", r"$\hat{h}(x)$")},
        'xlabel': 'x'
    },
    'sin(jagged_x)': {
        'output_label': 'contr-jaggedsin',
        'parser': sinfreq_parser,
        'coeff_names': ['a(t)'],
        'true_coeff_funcs': {'a(t)': t_fun},
        'coeff_labels': {'a(t)': (r'$a(t)$', r'$\hat{a}(t)$')},
        'xlabel': 'x'
    }
}

VAR_NAMES = ['x', 'y', 'z', 'w']    # variable names in sympy expressions generated from weights
PLOT_COLORS = [('0.7', 'blue')]
PLOT_COLORS_LATENT = [('0.7', 'blue'), ('#ccebc5', '#019E04'), ('#fbb4ae', '#e41a1c'), ('blue', 'orange')]


class Plotter:
    """Parent plotter object"""
    def __init__(self, arch_type, results_dir, func_name, trial_num, figures_dir, activation_funcs=None, var_names=None,
                 info_dict=None, font_size=20, extrapolation_font_size=16, title_font_size=20, legend_font_size=None,
                 extrapolation_colors=None, coeff_colors=None):
        """
        arch_type: architecture type, either 'para' or 'stack'
        results_dir: directory of results, probably results/benchmark-../test_..
        func_name: name of function (name of directory for function in results_dir)
        trial_num: the trial to plot, probably the best trial
        figures_dir: directory in which to save the figures
        activation_funcs: activation functions in EQL network, default ACTIVATION_FUNCS
        var_names: variable names for sympy expressions (doesn't matter what these are, as long as the same ones are used every time), default VAR_NAMES
        info_dict: default the appropriate FUNC_INFO_DICT
        font_size: font size for plots (besides extrapolation plots)
        extrapolation_font_size: font size for extrapolation plots
        title_font_size: font size for plot titles
        legend_font_size: font size for legends, default equal to font_size
        extrapolation_colors: colors used for extrapolation plot, default true=blue and predicted=orange
        coeff_colors: colors used for coefficient plots, default PLOT_COLORS
        """
        self.arch_type = arch_type
        self.results_dir = results_dir
        self.func_name = func_name
        self.trial_num = trial_num
        self.figures_dir = figures_dir

        if activation_funcs:
            self.activation_funcs = activation_funcs
        else:
            self.activation_funcs = ACTIVATION_FUNCS
        if var_names:
            self.var_names = var_names
        else:
            self.var_names = VAR_NAMES
        if info_dict:
            self.info_dict = info_dict
        else:
            self.info_dict = FUNC_INFO_DICTS[func_name]

        self.font_size = font_size
        self.extrapolation_font_size = extrapolation_font_size
        self.title_font_size = title_font_size

        if legend_font_size:
            self.legend_font_size = legend_font_size
        else:
            self.legend_font_size = font_size

        if extrapolation_colors:
            self.extrapolation_colors = extrapolation_colors
        else:
            self.extrapolation_colors = PLOT_COLORS[0]
        if coeff_colors:
            self.coeff_colors = coeff_colors
        else:
            self.coeff_colors = PLOT_COLORS_LATENT

    def get_model_expr_list(self, get_all=True):
        """
        Get a list of sympy expressions from model weights.
        get_all: False -> get expressions for 4 timesteps, True -> get expressions for all timesteps
        """
        func_name = self.func_name.replace("*", "_")
        with open('{}/{}/trial{}.pickle'.format(self.results_dir,func_name, self.trial_num), 'rb') as f:
            results = pickle.load(f)
        weights = results['weights']

        model_expr_list = []
        if get_all:
            for i in tqdm(range(weights[0].shape[0])):
                expr_idx = i
                if self.arch_type == 'para':
                    cur_weights = [w[expr_idx] for w in weights[:-1]] + [weights[-1]]
                elif self.arch_type == 'stack':
                    cur_weights = [w[expr_idx] for w in weights]
                else:
                    raise ArgumentError
                model_expr_list += [(expr_idx, pretty_print.network(cur_weights, self.activation_funcs, self.var_names[:weights[0].shape[1]]))]
        else:
            for i in tqdm(range(4)):
                expr_idx = (weights[0].shape[0] * i) // 4
                if self.arch_type == 'para':
                    cur_weights = [w[expr_idx] for w in weights[:-1]] + [weights[-1]]
                elif self.arch_type == 'stack':
                    cur_weights = [w[expr_idx] for w in weights]
                else:
                    raise ArgumentError
                model_expr_list += [(expr_idx, pretty_print.network(cur_weights, self.activation_funcs, self.var_names[:weights[0].shape[1]]))]
        
        return model_expr_list

    def get_expr_list(self, get_all=True):
        """
        Get a list of the true learned expressions (by unnormalizing variables in the expression from model weights).
        get_all: False -> get expressions for 4 timesteps, True -> get expressions for all timesteps
        """
        raise NotImplementedError("must override get_expr_list")

    def save_expr_list(self, get_all=True):
        """
        Save a list of the true learned expressions.
        get_all: False -> save expressions for 4 timesteps, True -> save expressions for all timesteps
        """
        expr_list = self.get_expr_list(get_all)

        with open('{}/{}-{}-exprlist.pickle'.format(self.figures_dir, self.arch_type, self.info_dict['output_label']), 'wb') as f:
            pickle.dump(expr_list, f)
        
        return expr_list

    def load_expr_list(self, get_all=True):
        """
        Load a list of the true learned expressions from a pickle file containing expressions for all timesteps.
        get_all: False -> load expressions for 4 timesteps, True -> load expressions for all timesteps
        """
        with open('{}/{}-{}-exprlist.pickle'.format(self.figures_dir, self.arch_type, self.info_dict['output_label']), 'rb') as f:
            expr_list = pickle.load(f)

        if get_all:
            return expr_list
        else:
            return [expr_list[(len(expr_list) * (4*i+1)) // 16] for i in range(4)]

    def get_t_batch(self, get_all=True):
        """
        Load the discrete values of t for a particular experiment.
        get_all: False -> load t values for 4 timesteps, True -> load t values for all timesteps
        """
        func_name = self.func_name.replace("*", "_")
        t_batch = np.load("{}/{}/t_batch_train.npy".format(self.results_dir, func_name)).flatten()

        if get_all:
            return [t for t in t_batch]
        else:
            return [t_batch[(len(t_batch) * (4*i+1)) // 16] for i in range(4)]

    def get_sympy_expr(self, expr):
        """Get a sympy expression from a string."""
        sympy_expr = sympy.parsing.sympy_parser.parse_expr(str(expr))
        sympy_expr = sympy.simplify(sympy.expand(sympy_expr))

        return sympy_expr

    def get_coeffs(self):
        """Get a dictionary of lists of coefficients from sympy expressions."""
        expr_list = self.load_expr_list()
        coeff_lists = [self.info_dict['parser'](self.get_sympy_expr(expr)) for expr in expr_list]

        coeff_dict = {coeff_name: list(coeff_list) for coeff_name, coeff_list in zip(self.info_dict['coeff_names'], zip(*coeff_lists))}

        return coeff_dict

    def get_training_time(self):
        """Get the training time of the model."""
        with open('{}/{}/trial{}.pickle'.format(self.results_dir, self.func_name, self.trial_num), 'rb') as f:
            results = pickle.load(f)
        training_time = results['training_time']

        return training_time

    def print_mse(self):
        func_name = self.func_name.replace("*", "_")
        with open('{}/{}/trial{}.pickle'.format(self.results_dir, func_name, self.trial_num), 'rb') as f:
            results = pickle.load(f)
        error_list = results['error_list']
        error_test_list = results['error_test']
        print(min(error_list))
        print(error_test_list[np.argmin(error_list)])
        print(min(error_test_list))

    def plot_mse(self):
        plt.rc('font', size=self.extrapolation_font_size)
        plt.rc('legend', fontsize=self.legend_font_size)

        with open('{}/{}/trial{}.pickle'.format(self.results_dir, self.func_name, self.trial_num), 'rb') as f:
            results = pickle.load(f)
        error_list = results['error_list']
        error_test_list = results['error_test']

        # forgot to reset error_list between trials
        if self.arch_type == 'para':
            error_list = error_list[-13:]
            error_test_list = error_test_list[-13:]
        # forgot to reset error_test_list between trials
        elif self.arch_type == 'stack':
            error_list = error_list[-7:]
            error_test_list = error_test_list[-7:]

        fig, ax = plt.subplots()

        ax.plot([i/(len(error_test_list)-1) for i in range(len(error_test_list))], error_list, label='train', color=self.extrapolation_colors[0])
        ax.plot([i/(len(error_test_list)-1) for i in range(len(error_test_list))], error_test_list, label='test', color=self.extrapolation_colors[1])
        ax.set_xlabel('epoch')
        ax.set_ylabel('mse')
        ax.set_yscale('log') # log scale
        ax.legend(bbox_to_anchor=(1.05,1), loc="upper left", markerscale=4)

        fig.savefig('{}/{}-{}-mse.png'.format(self.figures_dir, self.arch_type, self.info_dict['output_label']), bbox_inches='tight')
        plt.close(fig)


class ContrPlotter(Plotter):
    def get_expr_list(self, get_all=True):
        model_expr_list = self.get_model_expr_list(get_all)

        x_train = np.load('{}/{}/x_train.npy'.format(self.results_dir, self.func_name))
        y_train = np.load('{}/{}/y_train.npy'.format(self.results_dir, self.func_name))

        x_train_max = np.amax(x_train, axis=(0,1), keepdims=True)
        x_train_min = np.amin(x_train, axis=(0,1), keepdims=True)
        x_train_scale = np.maximum(x_train_max, -1*x_train_min)

        y_train_max = np.amax(y_train, axis=1, keepdims=True)
        y_train_min = np.amin(y_train, axis=1, keepdims=True)
        y_train_range = y_train_max-y_train_min

        for i in range(y_train_range.shape[0]):
            if y_train_range[i][0][0] == 0:
                y_train_range[i] = 1

        var_symbs = [sympy.Symbol(var_name) for var_name in self.var_names][:x_train.shape[2]]

        expr_list = [sympy.expand(y_train_range[expr_idx, 0, 0] * model_expr + y_train_min[expr_idx, 0, 0]) for expr_idx, model_expr in model_expr_list]
        for i in range(len(expr_list)):
            for j, var_symb in enumerate(var_symbs):
                expr_list[i] = sympy.expand(expr_list[i].subs(var_symb, var_symb/x_train_scale[0, 0, j]))

        return expr_list

    def plot_extrapolation(self, tplot=None):
        """Make extrapolation plots."""
        plt.rc('font', size=self.extrapolation_font_size-4)
        plt.rc('legend', fontsize=self.legend_font_size)

        func_name = self.func_name.replace("*", "_")
        x = np.load("{}/{}/x_test.npy".format(self.results_dir, func_name))
        y = np.load("{}/{}/y_test.npy".format(self.results_dir, func_name))
        t_batch = np.load("{}/{}/t_batch_test.npy".format(self.results_dir, func_name))

        y_hat = np.load("{}/{}/y_hat_{}.npy".format(self.results_dir, func_name, self.trial_num))

        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, sharex=True, sharey=True)
        for i, ax in enumerate([ax0, ax1, ax2, ax3]):
            if tplot is None:
                expr_idx = (t_batch.shape[0] * (4*i+1)) // 16
            else:
                expr_idx = np.argmin(np.abs(tplot[i] - t_batch))
            data_arr = list(zip(x[expr_idx], y[expr_idx], y_hat[expr_idx]))
            data_arr.sort()
            x_sorted, y_sorted, y_hat_sorted = zip(*data_arr)
            ax.plot(x_sorted, y_sorted, label='True', color=self.extrapolation_colors[0], zorder=0, linewidth=3)
            # ax.scatter(x_sorted, y_hat_sorted, s=1, label='pred', color=self.extrapolation_colors[1], zorder=4)
            ax.plot(x_sorted, y_hat_sorted, label='pred', color=self.extrapolation_colors[1], zorder=4, linewidth=4, linestyle=":")
            ax.axvspan(-7, -3, color='red', alpha=0.1)
            ax.axvspan(3, 7, color='red', alpha=0.1)
            ax.set_xlim(-5.25, 5.25)
            ax.set_title('t={}'.format(round(t_batch[expr_idx].item(), 4)))
            if i >= 2:
                ax.set_xlabel('x')
            if i % 2 == 0:
                ax.set_ylabel('y')
            if i == 1:
                ax.legend(bbox_to_anchor=(1.05,1), loc="upper left", markerscale=8)
        # fig.suptitle('{}'.format(self.func_name), fontsize=self.title_font_size)
        plt.subplots_adjust(bottom=0.15, top=0.85, left=0.15, right=0.95, hspace=0.3)
        # fig.savefig('{}/{}-{}-extrapolation.png'.format(self.figures_dir, self.arch_type, self.info_dict['output_label']), bbox_inches='tight')
        # plt.close(fig)

    def plot_coeffs(self):
        """Make coefficient plots."""
        plt.rc('font', size=self.font_size)
        plt.rc('legend', fontsize=self.legend_font_size)

        func_name = self.func_name.replace("*", "_")
        t_batch = np.load("{}/{}/t_batch_test.npy".format(self.results_dir, func_name))
        coeff_dict = self.get_coeffs()

        fig, ax = plt.subplots()
        for (var, coeffs), (true_color, pred_color) in zip(coeff_dict.items(), self.coeff_colors):
            ax.plot(t_batch, self.info_dict['true_coeff_funcs'][var](t_batch), label=self.info_dict['coeff_labels'][var][0],
                    color=true_color, zorder=0, linewidth=3)
            # ax.scatter(t_batch, coeffs, s=2, label=self.info_dict['coeff_labels'][var][1], color=pred_color, zorder=4)
            ax.plot(t_batch, coeffs, linestyle=":", label=self.info_dict['coeff_labels'][var][1], color=pred_color,
                    zorder=4, linewidth=4)
            break
        ax.set_xlabel(self.info_dict['xlabel'])
        ax.set_ylabel('a(t)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=8)
        # ax.set_title(self.func_name, fontsize=self.title_font_size)
        # plt.subplots_adjust(bottom=0.15, hspace=0.3)
        # fig.savefig('{}/{}-{}-coeffs.png'.format(self.figures_dir, self.arch_type, self.info_dict['output_label']), bbox_inches='tight')
        # plt.close(fig)

    def plot_coeffs_subplot(self):
        """Make coefficient plots."""
        plt.rc('font', size=self.font_size)
        plt.rc('legend', fontsize=self.legend_font_size)

        func_name = self.func_name.replace("*", "_")
        t_batch = np.load("{}/{}/t_batch_test.npy".format(self.results_dir, func_name))
        coeff_dict = self.get_coeffs()

        fig, ((ax0, ax1)) = plt.subplots(1, 2)
        fig.set_figwidth(10)
        for i, ax in enumerate([ax0, ax1]):
            (var, coeffs) = list(coeff_dict.items())[i]

            ax.plot(t_batch, self.info_dict['true_coeff_funcs'][var](t_batch), label="True",
                    color=self.extrapolation_colors[0], zorder=0, linewidth=3)
            # ax.scatter(t_batch, coeffs, s=2, label=self.info_dict['coeff_labels'][var][1], color=pred_color, zorder=4)
            ax.plot(t_batch, coeffs, linestyle=":", label="SEQL",
                    color=self.extrapolation_colors[1],
                    zorder=4, linewidth=4)
            ax.set_ylabel(self.info_dict['coeff_names'][i])
            ax.set_xlabel(self.info_dict['xlabel'])
        # plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=8)
        # ax.set_title(self.func_name, fontsize=self.title_font_size)
        # plt.subplots_adjust(bottom=0.15, hspace=0.3)
        # fig.savefig('{}/{}-{}-coeffs.png'.format(self.figures_dir, self.arch_type, self.info_dict['output_label']), bbox_inches='tight')
        # plt.close(fig)


class DiffEqPlotter(Plotter):
    def get_expr_list(self, get_all=True):
        model_expr_list = self.get_model_expr_list(get_all)

        x_train = np.load('{}/{}/x_train.npy'.format(self.results_dir, self.func_name))
        y_train = np.load('{}/{}/y_train.npy'.format(self.results_dir, self.func_name))

        x_train_max = np.amax(x_train, axis=(0,1), keepdims=True)
        x_train_min = np.amin(x_train, axis=(0,1), keepdims=True)
        x_train_scale = np.maximum(x_train_max, -1*x_train_min)

        y_train_max = np.amax(y_train, axis=1, keepdims=True)
        y_train_min = np.amin(y_train, axis=1, keepdims=True)
        y_train_range = y_train_max-y_train_min

        for i in range(y_train_range.shape[0]):
            if y_train_range[i][0][0] == 0:
                y_train_range[i] = 1

        var_symbs = [sympy.Symbol(var_name) for var_name in self.var_names][:x_train.shape[2]]

        expr_list = [sympy.expand(y_train_range[expr_idx, 0, 0] * model_expr + y_train_min[expr_idx, 0, 0]) for expr_idx, model_expr in model_expr_list]
        for i in range(len(expr_list)):
            for j, var_symb in enumerate(var_symbs):
                expr_list[i] = sympy.expand(expr_list[i].subs(var_symb, var_symb/x_train_scale[0, 0, j]))

        return expr_list

    def plot_colormap2(self):
        """Make colormap plots."""
        plt.rc('font', size=self.font_size)
        plt.rc('legend', fontsize=self.legend_font_size)

        x = np.load("{}/{}/x_test.npy".format(self.results_dir, self.func_name))
        y = np.load("{}/{}/y_test.npy".format(self.results_dir, self.func_name))
        t_batch = np.load("{}/{}/t_batch_test.npy".format(self.results_dir, self.func_name))

        y_hat = np.load("{}/{}/y_hat_{}.npy".format(self.results_dir, self.func_name, self.trial_num))
        y_hat_plot = y_hat[:, :, 0]
        y_plot = y[:, :, 0]

        if self.func_name == 'addiff':
            y_hat_plot = y_hat_plot.T
            y_plot = y_plot.T
            extent = [-5, 5, 0, 5]
            vmin, vmax = -3, 3
        elif self.func_name == 'burgers':
            extent = [-8, 8, 0, 10]
            vmin, vmax = -1.5, 1.5

        fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True, figsize=(14, 5))

        im0 = ax0.imshow(y_hat_plot, cmap='viridis', aspect='auto', extent=extent, vmin=vmin, vmax=vmax, origin='lower')
        im2 = ax1.imshow(y_hat_plot-y_plot, cmap='inferno', aspect='auto', extent=extent, origin='lower')

        ax0.set_title('Prediction')
        ax1.set_title('Error')

        # ax0.set_xlabel('x')
        ax0.set_ylabel('t')
        ax0.set_xlabel('x')
        # ax1.set_ylabel('t')
        ax1.set_xlabel('x')
        # ax1.set_ylabel('t')

        plt.colorbar(im0, ax=ax0, location='right')
        plt.colorbar(im2, ax=ax1, location='right')

        # fig.suptitle('{}'.format(self.func_name), fontsize=self.title_font_size)
        plt.subplots_adjust(bottom=0.15, top=0.85, left=0.15, right=0.95, hspace=0.3, wspace=0.3)
        fig.savefig('{}/{}-{}-colormap.png'.format(self.figures_dir, self.arch_type, self.info_dict['output_label']), bbox_inches='tight')
        plt.close(fig)

    def plot_colormap(self):
        """Make colormap plots."""
        plt.rc('font', size=self.font_size)
        plt.rc('legend', fontsize=self.legend_font_size)

        x = np.load("{}/{}/x_test.npy".format(self.results_dir, self.func_name))
        y = np.load("{}/{}/y_test.npy".format(self.results_dir, self.func_name))
        t_batch = np.load("{}/{}/t_batch_test.npy".format(self.results_dir, self.func_name))

        y_hat = np.load("{}/{}/y_hat_{}.npy".format(self.results_dir, self.func_name, self.trial_num))
        y_hat_plot = y_hat[:, :, 0]
        y_plot = y[:, :, 0]

        if self.func_name == 'addiff':
            y_hat_plot = y_hat_plot.T
            y_plot = y_plot.T
            extent = [-5, 5, 0, 5]
            vmin, vmax = -3, 3
        elif self.func_name == 'burgers':
            extent = [-8, 8, 0, 10]
            vmin, vmax = -1.5, 1.5

        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, sharex=True, sharey=True)

        im0 = ax0.imshow(y_hat_plot, cmap='viridis', aspect='auto', extent=extent, vmin=vmin, vmax=vmax, origin='lower')
        im1 = ax1.imshow(y_plot, cmap='viridis', aspect='auto', extent=extent, vmin=vmin, vmax=vmax, origin='lower')
        im2 = ax2.imshow(y_hat_plot-y_plot, cmap='inferno', aspect='auto', extent=extent, origin='lower')

        ax0.set_title('Prediction')
        ax1.set_title('true')
        ax2.set_title('error')
        ax3.axis('off')

        # ax0.set_xlabel('x')
        ax0.set_ylabel('t')
        ax1.set_xlabel('x')
        # ax1.set_ylabel('t')
        ax2.set_xlabel('x')
        ax2.set_ylabel('t')

        plt.colorbar(im0, ax=ax3, location='right')
        plt.colorbar(im2, ax=ax3, location='left')

        # fig.suptitle('{}'.format(self.func_name), fontsize=self.title_font_size)
        plt.subplots_adjust(bottom=0.15, top=0.85, left=0.15, right=0.95, hspace=0.3, wspace=0.3)
        fig.savefig('{}/{}-{}-colormap.png'.format(self.figures_dir, self.arch_type, self.info_dict['output_label']), bbox_inches='tight')
        plt.close(fig)

    def plot_colormap_values(self):
        """Make the true and predicted colormap plots (top row in plot_colormap)."""
        plt.rc('font', size=self.font_size)
        plt.rc('legend', fontsize=self.legend_font_size)

        x = np.load("{}/{}/x_test.npy".format(self.results_dir, self.func_name))
        y = np.load("{}/{}/y_test.npy".format(self.results_dir, self.func_name))
        t_batch = np.load("{}/{}/t_batch_test.npy".format(self.results_dir, self.func_name))

        y_hat = np.load("{}/{}/y_hat_{}.npy".format(self.results_dir, self.func_name, self.trial_num))
        y_hat_plot = y_hat[:, :, 0]
        y_plot = y[:, :, 0]

        if self.func_name == 'addiff':
            y_hat_plot = y_hat_plot.T
            y_plot = y_plot.T
            extent = [-5, 5, 0, 5]
            vmin, vmax = -3, 3
        elif self.func_name == 'burgers':
            extent = [-8, 8, 0, 10]
            vmin, vmax = -1.5, 1.5

        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(9.6, 2.4))

        im0 = ax0.imshow(y_hat_plot, cmap='viridis', aspect='auto', extent=extent, vmin=vmin, vmax=vmax, origin='lower')
        im1 = ax1.imshow(y_plot, cmap='viridis', aspect='auto', extent=extent, vmin=vmin, vmax=vmax, origin='lower')

        ax0.set_title('EQL')
        ax1.set_title('true')
        ax2.axis('off')

        ax0.set_xlabel('x')
        ax0.set_ylabel('t')
        ax1.set_xlabel('x')
        # ax1.set_ylabel('t')

        plt.colorbar(im0, ax=ax2, location='left')

        # fig.suptitle('{}'.format(self.func_name), fontsize=self.title_font_size)
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        fig.savefig('{}/{}-{}-colormap-values.png'.format(self.figures_dir, self.arch_type, self.info_dict['output_label']), bbox_inches='tight')
        plt.close(fig)

    def plot_colormap_errors(self):
        """Make the error colormap plot (bottom row in plot_colormap)."""
        plt.rc('font', size=self.font_size)
        plt.rc('legend', fontsize=self.legend_font_size)

        x = np.load("{}/{}/x_test.npy".format(self.results_dir, self.func_name))
        y = np.load("{}/{}/y_test.npy".format(self.results_dir, self.func_name))
        t_batch = np.load("{}/{}/t_batch_test.npy".format(self.results_dir, self.func_name))

        y_hat = np.load("{}/{}/y_hat_{}.npy".format(self.results_dir, self.func_name, self.trial_num))
        y_hat_plot = y_hat[:, :, 0]
        y_plot = y[:, :, 0]

        if self.func_name == 'addiff':
            y_hat_plot = y_hat_plot.T
            y_plot = y_plot.T
            extent = [-5, 5, 0, 5]
            vmin, vmax = -3, 3
        elif self.func_name == 'burgers':
            extent = [-8, 8, 0, 10]
            vmin, vmax = -1.5, 1.5

        fig, (ax0, ax1) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(7.2, 2.4))

        im0 = ax0.imshow(y_hat_plot-y_plot, cmap='inferno', aspect='auto', extent=extent, origin='lower')

        ax0.set_title('error')
        ax1.axis('off')

        ax0.set_xlabel('x')
        ax0.set_ylabel('t')

        plt.colorbar(im0, ax=ax1, location='left')

        # fig.suptitle('{}'.format(self.func_name), fontsize=self.title_font_size)
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        fig.savefig('{}/{}-{}-colormap-errors.png'.format(self.figures_dir, self.arch_type, self.info_dict['output_label']), bbox_inches='tight')
        plt.close(fig)

    def plot_coeffs(self):
        "Make coefficient plots."
        plt.rc('font', size=self.font_size)
        plt.rc('legend', fontsize=self.legend_font_size)

        t_batch = np.load("{}/{}/t_batch_test.npy".format(self.results_dir, self.func_name))
        coeff_dict = self.get_coeffs()

        fig, ax = plt.subplots()
        for (var, coeffs), (true_color, pred_color) in zip(coeff_dict.items(), self.coeff_colors):
            ax.plot(t_batch, self.info_dict['true_coeff_funcs'][var](t_batch), label=self.info_dict['coeff_labels'][var][0], color=true_color, zorder=0, linewidth=3)
            # ax.scatter(t_batch, coeffs, s=2, label=self.info_dict['coeff_labels'][var][1], color=pred_color, zorder=4)
            ax.plot(t_batch, coeffs, label=self.info_dict['coeff_labels'][var][1], color=pred_color, zorder=4, linestyle=":", linewidth=3)
        ax.set_xlabel(self.info_dict['xlabel'])
        # ax.set_ylabel('predicted coefficients')
        ax.legend(bbox_to_anchor=(1.05,1), loc="upper left", markerscale=4)
        # ax.set_title(self.func_name, fontsize=self.title_font_size)
        ax.set_title('Parametric Coefficients')
        plt.subplots_adjust(bottom=0.15, hspace=0.3)
        fig.savefig('{}/{}-{}-coeffs.png'.format(self.figures_dir, self.arch_type, self.info_dict['output_label']), bbox_inches='tight')
        plt.close(fig)

    def print_coeff_errors(self):
        """Make plots of absolute coefficient errors."""
        t_batch = np.load("{}/{}/t_batch_test.npy".format(self.results_dir, self.func_name))
        coeff_dict = self.get_coeffs()

        errors = []

        for (var, coeffs), (_, pred_color) in zip(coeff_dict.items(), self.coeff_colors):
            errors.append(np.array(coeffs) - np.squeeze(self.info_dict['true_coeff_funcs'][var](t_batch)))

        print(np.mean(np.array(errors)**2))


class ConvPlotter(Plotter):
    def __init__(self, arch_type, results_dir, func_name, trial_num, figures_dir, activation_funcs=None, var_names=None, info_dict=None, font_size=20, extrapolation_font_size=16, title_font_size=20, legend_font_size=None, extrapolation_colors=None, coeff_colors=None, relu_dir=None, relu_trial_num=None):
        super().__init__(arch_type, results_dir, func_name, trial_num, figures_dir, activation_funcs, var_names, info_dict, font_size, extrapolation_font_size, title_font_size, legend_font_size, extrapolation_colors, coeff_colors)
        self.relu_dir = relu_dir
        self.relu_trial_num = relu_trial_num

    def get_expr_list(self, get_all=True):
        model_expr_list = self.get_model_expr_list(get_all)

        y_train = np.load('{}/{}/y_train.npy'.format(self.results_dir, self.func_name))

        y_train_max = np.amax(y_train, axis=1, keepdims=True)
        y_train_min = np.amin(y_train, axis=1, keepdims=True)
        y_train_range = y_train_max-y_train_min

        for i in range(y_train_range.shape[0]):
            if y_train_range[i][0][0] == 0:
                y_train_range[i] = 1

        expr_list = [sympy.expand(y_train_range[expr_idx, 0, 0] * model_expr + y_train_min[expr_idx, 0, 0]) for expr_idx, model_expr in model_expr_list]

        return expr_list

    def plot_extrapolation(self):
        "Make extrapolation plots."
        plt.rc('font', size=self.extrapolation_font_size)
        plt.rc('legend', fontsize=self.legend_font_size)

        x = np.load("{}/{}/x_test.npy".format(self.results_dir, self.func_name))
        y = np.load("{}/{}/y_test.npy".format(self.results_dir, self.func_name))
        t_batch = np.load("{}/{}/t_batch_test.npy".format(self.results_dir, self.func_name))

        y_hat = np.load("{}/{}/y_hat_{}.npy".format(self.results_dir, self.func_name, self.trial_num))

        x = x[:, :, 1] - x[:, :, 0]

        if self.relu_dir is not None and self.relu_trial_num is not None:
            x_relu = np.load("{}/{}/x_test.npy".format(self.relu_dir, self.func_name))
            y_hat_relu = np.load("{}/{}/y_hat_{}.npy".format(self.relu_dir, self.func_name, self.relu_trial_num))
            x_relu = x_relu[:, :, 1] - x_relu[:, :, 0]
        else:
            x_relu = None
            y_hat_relu = None

        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, sharex=True, sharey=True)
        for i, ax in enumerate([ax0, ax1, ax2, ax3]):
            expr_idx = (t_batch.shape[0] * (4*i+1)) // 16
            data_arr = list(zip(x[expr_idx], y[expr_idx], y_hat[expr_idx]))
            data_arr.sort()
            x_sorted, y_sorted, y_hat_sorted = zip(*data_arr)
            ax.plot(x_sorted, y_sorted, label='True', color=self.extrapolation_colors[0], zorder=0, linewidth=3)
            # ax.scatter(x_sorted, y_hat_sorted, s=1, label='EQL', color=self.extrapolation_colors[1], zorder=4)
            ax.plot(x_sorted, y_hat_sorted, label='EQL', color=self.extrapolation_colors[1], zorder=4, linestyle=":", linewidth=3)
            if x_relu is not None and y_hat_relu is not None:
                # ax.scatter(x_relu[expr_idx], y_hat_relu[expr_idx], s=1, label='ReLU', color='#8c510a', zorder=2)
                x_relu_sorted = sorted(x_relu[expr_idx])
                y_relu_sorted = [y for _, y in sorted(zip(x_relu[expr_idx], y_hat_relu[expr_idx]))]
                ax.plot(x_relu_sorted, y_relu_sorted, label='ReLU', color='#8c510a', zorder=2, linestyle=":", linewidth=3)
            ax.axvspan(-8, -4, color='red', alpha=0.1)
            ax.axvspan(4, 8, color='red', alpha=0.1)
            ax.set_xlim(-6.25, 6.25)
            ax.set_title('t={}'.format(round(t_batch[expr_idx].item(), 4)))
            if i >= 2:
                ax.set_xlabel(r'$\psi_2-\psi_1$')
            if i % 2 == 0:
                ax.set_ylabel('y')
            if i == 1:
                ax.legend(bbox_to_anchor=(1.05,1), loc="upper left", markerscale=8)
        # fig.suptitle('{}'.format(self.func_name), fontsize=self.title_font_size)
        plt.subplots_adjust(bottom=0.15, top=0.85, left=0.15, right=0.95, hspace=0.3)
        fig.savefig('{}/{}-{}-extrapolation.png'.format(self.figures_dir, self.arch_type, self.info_dict['output_label']),bbox_inches='tight')
        plt.close(fig)

    def plot_coeffs(self):
        "Make coefficient plots."
        plt.rc('font', size=self.font_size)
        plt.rc('legend', fontsize=self.legend_font_size)

        t_batch = np.load("{}/{}/t_batch_test.npy".format(self.results_dir, self.func_name))
        coeff_dict = self.get_coeffs()

        fig, ax = plt.subplots()
        linestyles = ['--', '-.', ':']
        for (var, coeffs), (_, pred_color), linestyle in zip(coeff_dict.items(), self.coeff_colors, linestyles):
            # ax.scatter(t_batch, coeffs, s=2, label=self.info_dict['coeff_labels'][var][1], color=pred_color)
            ax.plot(t_batch, coeffs, label=self.info_dict['coeff_labels'][var][1], color=pred_color, linestyle=linestyle, linewidth=3)
        ax.set_xlabel(self.info_dict['xlabel'])
        ax.set_ylabel('predicted coefficients')
        ax.legend(bbox_to_anchor=(1.05,1), loc="upper left", markerscale=8)
        # ax.set_title(self.func_name, fontsize=self.title_font_size)
        plt.subplots_adjust(bottom=0.15, hspace=0.3)
        fig.savefig('{}/{}-{}-coeffs.png'.format(self.figures_dir, self.arch_type, self.info_dict['output_label']), bbox_inches='tight')
        plt.close(fig)

    def plot_latent(self):
        "Make latent-space plots."
        plt.rc('font', size=self.font_size)
        plt.rc('legend', fontsize=self.legend_font_size)

        x = np.load('{}/{}/x_test.npy'.format(self.results_dir, self.func_name))
        z = np.load('{}/{}/l_hat_{}.npy'.format(self.results_dir, self.func_name, self.trial_num))

        x_plot = x[[(x.shape[0] * (4*i+1)) // 16 for i in range(4)]]
        z_plot = z[[(z.shape[0] * (4*i+1)) // 16 for i in range(4)]]
        x_plot.flatten()
        z_plot.flatten()

        fig, ax = plt.subplots()
        ax.scatter(x, z, s=2, color='black')
        ax.set_xlabel(r'true parameter $\psi$')
        ax.set_ylabel(r'latent parameter $\hat{z}$')

        # ax.set_title('{} latent encoding'.format(self.func_name), fontsize=self.title_font_size)
        plt.subplots_adjust(bottom=0.15, hspace=0.3)
        fig.savefig('{}/{}-{}-latent.png'.format(self.figures_dir, self.arch_type, self.info_dict['output_label']), bbox_inches='tight')
        plt.close(fig)

    def get_latent_scaling(self):
        "Get the scaling of the latent variable versus the original variable through a linear regression."
        x = np.load('{}/{}/x_test.npy'.format(self.results_dir, self.func_name))
        z = np.load('{}/{}/l_hat_{}.npy'.format(self.results_dir, self.func_name, self.trial_num))

        x_scale = x[[(x.shape[0] * (4*i+1)) // 16 for i in range(4)]]
        z_scale = z[[(z.shape[0] * (4*i+1)) // 16 for i in range(4)]]
        x_scale = np.expand_dims(x_scale.flatten(), axis=-1)
        z_scale = z_scale.flatten()

        return LinearRegression().fit(x_scale, z_scale).coef_.item()

    def load_orig_expr_list(self, get_all=True):
        "Get the true learned equations (by undoing the latent space scaling)."
        expr_list = self.load_expr_list(get_all)
        scale = self.get_latent_scaling()

        x_train = np.load('{}/{}/x_train.npy'.format(self.results_dir, self.func_name))
        var_symbs = [sympy.Symbol(var_name) for var_name in self.var_names][:x_train.shape[2]]

        for i in range(len(expr_list)):
            for var_symb in var_symbs:
                expr_list[i] = sympy.expand(expr_list[i].subs(var_symb, var_symb*scale))

        return expr_list
    
    def plot_relu_latent(self):
        "Make latent-space plots of the covolutional ReLU network."
        plt.rc('font', size=self.font_size)
        plt.rc('legend', fontsize=self.legend_font_size)

        x = np.load('{}/{}/x_test.npy'.format(self.relu_dir, self.func_name))
        z = np.load('{}/{}/l_hat_{}.npy'.format(self.relu_dir, self.func_name, self.relu_trial_num))

        x_plot = x[[(x.shape[0] * (4*i+1)) // 16 for i in range(4)]]
        z_plot = z[[(z.shape[0] * (4*i+1)) // 16 for i in range(4)]]
        x_plot.flatten()
        z_plot.flatten()

        fig, ax = plt.subplots()
        ax.scatter(x, z, s=2)
        ax.set_xlabel(r'true parameter $\psi$')
        ax.set_ylabel(r'latent parameter $\hat{z}$')

        # ax.set_title('{} latent encoding'.format(self.func_name), fontsize=self.title_font_size)
        plt.subplots_adjust(bottom=0.15, hspace=0.3)
        fig.savefig('{}/relu-{}-latent.png'.format(self.figures_dir, self.info_dict['output_label']), bbox_inches='tight')
        plt.close(fig)


class SchedulePlotter():
    def __init__(self, figures_dir, font_size=20, title_font_size=20):
        self.figures_dir = figures_dir
        self.font_size = font_size
        self.title_font_size = title_font_size

    def plot_lr(self):
        "Plot the learning rate schedule."
        plt.rc('font', size=self.font_size)
        ratios_list = [e/210 for e in range(210)]

        lr_list = [schedules.get_lr(1, e, 210) for e in range(210)]
        lr_fig, lr_ax = plt.subplots()
        lr_ax.plot(ratios_list, lr_list, color='k', linewidth=3)
        # lr_ax.set_title('learning rate schedule', fontsize=self.title_font_size)
        lr_ax.set_xlabel('epoch')
        lr_ax.set_ylabel('learning rate')
        plt.subplots_adjust(bottom=0.15, hspace=0.3)
        lr_fig.savefig('{}/lr.png'.format(self.figures_dir), bbox_inches='tight')
        plt.close(lr_fig)

    def plot_rw(self):
        """Plot the regularization weight schedule."""
        plt.rc('font', size=self.font_size)
        ratios_list = [e/210 for e in range(210)]

        rw_list = [schedules.get_rw(1, e, 210) for e in range(210)]
        rw_fig, rw_ax = plt.subplots()
        rw_ax.plot(ratios_list, rw_list, color='k', linewidth=3)
        # rw_ax.set_title('regularization weight schedule', fontsize=self.title_font_size)
        rw_ax.set_xlabel('epoch')
        rw_ax.set_ylabel('regularization weight')
        plt.subplots_adjust(bottom=0.15, hspace=0.3)
        rw_fig.savefig('{}/rw.png'.format(self.figures_dir), bbox_inches='tight')
        plt.close(rw_fig)
