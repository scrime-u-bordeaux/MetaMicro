o
    \~hS>  �                   @   s�  d Z ddlZddlZddlZddlZddlZddlmZm	Z	m
Z
mZ ddlmZ ddgZddddd	d
dddd�	de	ej dede	ej dedede	e dede	ej de	ej dedejfdd�Zddddddddddddd�de	ej dede	ej ded ed!e	e d"ed#ed$ed%ede	ej de	ej dedejfd&d�Zddddd'dddddd(�
de	ej de	ej de	e d e	e d%ed!e	e d"ed#ed$ede	ej deejef fd)d*�ZdS )+zSpectral feature extraction�    N)�Any�Optional�Union�
Collection)�	DTypeLike�melspectrogram�mfcci"V  �   �   �ortho)	�y�sr�S�n_mfcc�dct_type�norm�lifter�	mel_basis�
fft_windowr   r   r   r   r   r   r   r   r   �kwargs�returnc        	      	   K   s�   |du rt �td| |||d�|	���}tjj|d||d�dg d�dd�f }
|dkrSt�tjtj	dd| |
j
d	� | �}t jj||jdd
�}|
d|d |  9 }
|
S |dkrY|
S t �d|� d���)ac  Mel-frequency cepstral coefficients (MFCCs)

    .. warning:: If multi-channel audio input ``y`` is provided, the MFCC
        calculation will depend on the peak loudness (in decibels) across
        all channels.  The result may differ from independent MFCC calculation
        of each channel.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n,)] or None
        audio time series. Multi-channel is supported..
    sr : number > 0 [scalar]
        sampling rate of ``y``
    S : np.ndarray [shape=(..., d, t)] or None
        log-power Mel spectrogram
    n_mfcc : int > 0 [scalar]
        number of MFCCs to return
    dct_type : {1, 2, 3}
        Discrete cosine transform (DCT) type.
        By default, DCT type-2 is used.
    norm : None or 'ortho'
        If ``dct_type`` is `2 or 3`, setting ``norm='ortho'`` uses an ortho-normal
        DCT basis.
        Normalization is not supported for ``dct_type=1``.
    lifter : number >= 0
        If ``lifter>0``, apply *liftering* (cepstral filtering) to the MFCCs::
            M[n, :] <- M[n, :] * (1 + sin(pi * (n + 1) / lifter) * lifter / 2)
        Setting ``lifter >= 2 * n_mfcc`` emphasizes the higher-order coefficients.
        As ``lifter`` increases, the coefficient weighting becomes approximately linear.
    **kwargs : additional keyword arguments to `melspectrogram`
        if operating on time series input
    n_fft : int > 0 [scalar]
        length of the FFT window
    hop_length : int > 0 [scalar]
        number of samples between successive frames.
        See `librosa.stft`
    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by `window()`.
        The window will be of length `win_length` and then padded
        with zeros to match ``n_fft``.
        If unspecified, defaults to ``win_length = n_fft``.
    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
        see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a vector or array of length ``n_fft``
        .. see also:: `librosa.filters.get_window`
    center : boolean
        - If `True`, the signal ``y`` is padded so that frame
        ``t`` is centered at ``y[t * hop_length]``.
        - If `False`, then frame ``t`` begins at ``y[t * hop_length]``
    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.
    power : float > 0 [scalar]
        Exponent applied to the spectrum before calculating the melspectrogram when the input is a time signal,
        e.g. 1 for magnitude, 2 for power **(default)**, etc.
    **kwargs : additional keyword arguments for Mel filter bank parameters
    n_mels : int > 0 [scalar]
        number of Mel bands to generate
    fmin : float >= 0 [scalar]
        lowest frequency (in Hz)
    fmax : float >= 0 [scalar]
        highest frequency (in Hz).
        If `None`, use ``fmax = sr / 2.0``
    htk : bool [scalar]
        use HTK formula instead of Slaney
    dtype : np.dtype
        The data type of the output basis.
        By default, uses 32-bit (single-precision) floating point.

    Returns
    -------
    M : np.ndarray [shape=(..., n_mfcc, t)]
        MFCC sequence

    See Also
    --------
    melspectrogram
    scipy.fftpack.dct

    Examples
    --------
    Generate mfccs from a time series

    >>> y, sr = librosa.load(librosa.ex('libri1'))
    >>> librosa.feature.mfcc(y=y, sr=sr)
    array([[-565.919, -564.288, ..., -426.484, -434.668],
           [  10.305,   12.509, ...,   88.43 ,   90.12 ],
           ...,
           [   2.807,    2.068, ...,   -6.725,   -5.159],
           [   2.822,    2.244, ...,   -6.198,   -6.177]], dtype=float32)

    Using a different hop length and HTK-style Mel frequencies

    >>> librosa.feature.mfcc(y=y, sr=sr, hop_length=1024, htk=True)
    array([[-5.471e+02, -5.464e+02, ..., -4.446e+02, -4.200e+02],
           [ 1.361e+01,  1.402e+01, ...,  9.764e+01,  9.869e+01],
           ...,
           [ 4.097e-01, -2.029e+00, ..., -1.051e+01, -1.130e+01],
           [-1.119e-01, -1.688e+00, ..., -3.442e+00, -4.687e+00]],
          dtype=float32)

    Use a pre-computed log-power Mel spectrogram

    >>> S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
    ...                                    fmax=8000)
    >>> librosa.feature.mfcc(S=librosa.power_to_db(S))
    array([[-559.974, -558.449, ..., -411.96 , -420.458],
           [  11.018,   13.046, ...,   76.972,   80.888],
           ...,
           [   2.713,    2.379, ...,    1.464,   -2.835],
           [   2.712,    2.619, ...,    2.209,    0.648]], dtype=float32)

    Get more components

    >>> mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    Visualize the MFCC series

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
    ...                                x_axis='time', y_axis='mel', fmax=8000,
    ...                                ax=ax[0])
    >>> fig.colorbar(img, ax=[ax[0]])
    >>> ax[0].set(title='Mel spectrogram')
    >>> ax[0].label_outer()
    >>> img = librosa.display.specshow(mfccs, x_axis='time', ax=ax[1])
    >>> fig.colorbar(img, ax=[ax[1]])
    >>> ax[1].set(title='MFCC')

    Compare different DCT bases

    >>> m_slaney = librosa.feature.mfcc(y=y, sr=sr, dct_type=2)
    >>> m_htk = librosa.feature.mfcc(y=y, sr=sr, dct_type=3)
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> img1 = librosa.display.specshow(m_slaney, x_axis='time', ax=ax[0])
    >>> ax[0].set(title='RASTAMAT / Auditory toolbox (dct_type=2)')
    >>> fig.colorbar(img, ax=[ax[0]])
    >>> img2 = librosa.display.specshow(m_htk, x_axis='time', ax=ax[1])
    >>> ax[1].set(title='HTK-style (dct_type=3)')
    >>> fig.colorbar(img2, ax=[ax[1]])
    N)r   r   r   r   �����)�axis�typer   .)	r   �   r
   �   �   �   �   �
   �   r   r   )�dtype)�ndim�axesr
   zMFCC lifter=z must be a non-negative number� )�librosa�power_to_dbr   �scipy�fftpack�dct�np�sin�pi�aranger!   �util�	expand_tor"   �ParameterError)r   r   r   r   r   r   r   r   r   r   �M�LIr$   r$   �t/home/noemie/Documents/enseirb/stage_scrime/MetaMicro/meta_micro_de_base/scripts/ta_la_ti_li/modif_libro/spectral.pyr      s    �&i   i   �hannT�constantg       @)r   r   r   �n_fft�
hop_length�
win_length�window�center�pad_mode�powerr   r   r6   r7   r8   r9   r:   r;   r<   c                 K   s0   t | ||||	|||||d�
\}}t�|
|�}|S )a  Compute a mel-scaled spectrogram.

    If a spectrogram input ``S`` is provided, then it is mapped directly onto
    the mel basis by ``mel_f.dot(S)``.

    If a time-series input ``y, sr`` is provided, then its magnitude spectrogram
    ``S`` is first computed, and then mapped onto the mel scale by
    ``mel_f.dot(S**power)``.

    By default, ``power=2`` operates on a power spectrum.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        audio time-series. Multi-channel is supported.
    sr : number > 0 [scalar]
        sampling rate of ``y``
    S : np.ndarray [shape=(..., d, t)]
        spectrogram
    n_fft : int > 0 [scalar]
        length of the FFT window
    hop_length : int > 0 [scalar]
        number of samples between successive frames.
        See `librosa.stft`
    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by `window()`.
        The window will be of length `win_length` and then padded
        with zeros to match ``n_fft``.
        If unspecified, defaults to ``win_length = n_fft``.
    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a vector or array of length ``n_fft``
        .. see also:: `librosa.filters.get_window`
    center : boolean
        - If `True`, the signal ``y`` is padded so that frame
          ``t`` is centered at ``y[t * hop_length]``.
        - If `False`, then frame ``t`` begins at ``y[t * hop_length]``
    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.
    power : float > 0 [scalar]
        Exponent for the magnitude melspectrogram.
        e.g., 1 for energy, 2 for power, etc.
    **kwargs : additional keyword arguments for Mel filter bank parameters
    n_mels : int > 0 [scalar]
        number of Mel bands to generate
    fmin : float >= 0 [scalar]
        lowest frequency (in Hz)
    fmax : float >= 0 [scalar]
        highest frequency (in Hz).
        If `None`, use ``fmax = sr / 2.0``
    htk : bool [scalar]
        use HTK formula instead of Slaney
    norm : {None, 'slaney', or number} [scalar]
        If 'slaney', divide the triangular mel weights by the width of
        the mel band (area normalization).
        If numeric, use `librosa.util.normalize` to normalize each filter
        by to unit l_p norm. See `librosa.util.normalize` for a full
        description of supported norm values (including `+-np.inf`).
        Otherwise, leave all the triangles aiming for a peak value of 1.0
    dtype : np.dtype
        The data type of the output basis.
        By default, uses 32-bit (single-precision) floating point.

    Returns
    -------
    S : np.ndarray [shape=(..., n_mels, t)]
        Mel spectrogram

    See Also
    --------
    librosa.filters.mel : Mel filter bank construction
    librosa.stft : Short-time Fourier Transform

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> librosa.feature.melspectrogram(y=y, sr=sr)
    array([[3.837e-06, 1.451e-06, ..., 8.352e-14, 1.296e-11],
           [2.213e-05, 7.866e-06, ..., 8.532e-14, 1.329e-11],
           ...,
           [1.115e-05, 5.192e-06, ..., 3.675e-08, 2.470e-08],
           [6.473e-07, 4.402e-07, ..., 1.794e-08, 2.908e-08]],
          dtype=float32)

    Using a pre-computed power spectrogram would give the same result:

    >>> D = np.abs(librosa.stft(y))**2
    >>> S = librosa.feature.melspectrogram(S=D, sr=sr)

    Display of mel-frequency spectrogram coefficients, with custom
    arguments for mel filterbank construction (default is fmax=sr/2):

    >>> # Passing through arguments to the Mel filters
    >>> S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
    ...                                     fmax=8000)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> S_dB = librosa.power_to_db(S, ref=np.max)
    >>> img = librosa.display.specshow(S_dB, x_axis='time',
    ...                          y_axis='mel', sr=sr,
    ...                          fmax=8000, ax=ax)
    >>> fig.colorbar(img, ax=ax, format='%+2.0f dB')
    >>> ax.set(title='Mel-frequency spectrogram')
    �
r   r   r6   r7   r<   r8   r9   r:   r;   r   )�_spectrogramr*   �dot)r   r   r   r6   r7   r8   r9   r:   r;   r<   r   r   r   �melspecr$   r$   r3   r   �   s   |
�r   r=   c        
   
      C   s�   |dur |du s|d d |j d krd|j d d  }||fS |du r,t�d|� ���| du r5t�d��t�tj| ||||||d��| }||fS )a  Retrieve a magnitude spectrogram.

    This is primarily used in feature extraction functions that can operate on
    either audio time-series or spectrogram input.

    Parameters
    ----------
    y : None or np.ndarray
        If provided, an audio time series

    S : None or np.ndarray
        Spectrogram input, optional

    n_fft : int > 0
        STFT window size

    hop_length : int > 0
        STFT hop length

    power : float > 0
        Exponent for the magnitude spectrogram,
        e.g., 1 for energy, 2 for power, etc.

    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by ``window``.
        The window will be of length ``win_length`` and then padded
        with zeros to match ``n_fft``.

        If unspecified, defaults to ``win_length = n_fft``.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a vector or array of length ``n_fft``

        .. see also:: `filters.get_window`

    center : boolean
        - If ``True``, the signal ``y`` is padded so that frame
          ``t`` is centered at ``y[t * hop_length]``.
        - If ``False``, then frame ``t`` begins at ``y[t * hop_length]``

    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.

    Returns
    -------
    S_out : np.ndarray [dtype=np.float]
        - If ``S`` is provided as input, then ``S_out == S``
        - Else, ``S_out = |stft(y, ...)|**power``
    n_fft : int > 0
        - If ``S`` is provided, then ``n_fft`` is inferred from ``S``
        - Else, copied from input
    Nr
   r   r   z)Unable to compute spectrogram with n_fft=z6Input signal must be provided to compute a spectrogram)r6   r7   r8   r:   r9   r;   )�shaper%   r0   r*   �abs�stftr=   r$   r$   r3   r>   Y  s2   F������r>   )�__doc__r%   �numpyr*   r'   �scipy.signal�scipy.fftpack�typingr   r   r   r   �numpy.typingr   �__all__�ndarray�float�int�strr   �boolr   �tupler>   r$   r$   r$   r3   �<module>   s�   �	��������	�
��
� 7��������	�
�����
� ��������	�
���