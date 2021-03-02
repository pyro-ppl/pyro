# Copyright (c) 2014, Salesforce.com, Inc.  All rights reserved.
# Copyright (c) 2015-2016, Gamelan Labs, Inc.
# Copyright (c) 2016, Google, Inc.
# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# - Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# - Neither the name of Salesforce.com nor the names of its contributors
#   may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
This module computes the chi^2 survival function and the required
functions.
"""

import math
import sys


def log(x):
    if x > sys.float_info.min:
        value = math.log(x)
    else:
        value = -math.inf
    return value


def incomplete_gamma(x, s):
    r"""
    This function computes the incomplete lower gamma function
    using the series expansion:

    .. math::

       \gamma(x, s) = x^s \Gamma(s)e^{-x}\sum^\infty_{k=0}
                    \frac{x^k}{\Gamma(s + k + 1)}

    This series will converge strongly because the Gamma
    function grows factorially.

    Because the Gamma function does grow so quickly, we can
    run into numerical stability issues. To solve this we carry
    out as much math as possible in the log domain to reduce
    numerical error. This function matches the results from
    scipy to numerical precision.
    """
    if x < 0:
        return 1
    if x > 1e3:
        return math.gamma(s)
    log_gamma_s = math.lgamma(s)
    log_x = log(x)
    value = 0
    for k in range(100):
        log_num = (k + s) * log_x + (-x) + log_gamma_s
        log_denom = math.lgamma(k + s + 1)
        value += math.exp(log_num - log_denom)
    return value


def chi2sf(x, s):
    r"""
    This function returns the survival function of the chi^2
    distribution. The survival function is given as:

    .. math::
       1 - CDF

    where rhe chi^2 CDF is given as

    .. math::
       F(x; s) = \frac{ \gamma( x/2, s/2 ) }{ \Gamma(s/2) },

    with :math:`\gamma` is the incomplete gamma function defined above.
    Therefore, the survival probability is givne by:

    .. math::
       1 - \frac{ \gamma( x/2, s/2 ) }{ \Gamma(s/2) }.

    This function matches the results from
    scipy to numerical precision.
    """
    top = incomplete_gamma(x / 2, s / 2)
    bottom = math.gamma(s / 2)
    value = top / bottom
    return 1 - value
