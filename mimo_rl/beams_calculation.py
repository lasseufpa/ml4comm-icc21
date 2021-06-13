'''
Assumes ULA at base station and single-antenna users.
'''
import numpy as np

class AnalogBeamformer:
    def __init__(self, num_antenna_elements=32):
        self.num_antenna_elements = num_antenna_elements
        self.codebook = dft_codebook(self.num_antenna_elements)
        num_points = 500  # resolution for plotting the beams
        # dimension num_points x num_antenna_elements
        self.beams_for_plotting, self.angles_for_plotting = self.get_steered_factors(
            num_antenna_elements, num_points)

    def get_steered_factors(self, num_antenna_elements, num_points):
        # Calculate the steering factor
        # grid, in angular domain
        theta = np.linspace(-np.pi, np.pi, num_points)
        theta = theta[:, np.newaxis]
        arrayFactors = arrayFactorGivenAngleForULA(
            num_antenna_elements, theta)
        steeredArrayFactors = np.squeeze(
            np.matmul(arrayFactors, self.codebook))
        return steeredArrayFactors, theta

    def get_num_codevectors(self):
        return self.codebook.shape[0]

    def get_best_beam_index(self, H):
        EquivalentChannels = np.dot(np.squeeze(np.asarray(H)), self.codebook)
        bestIndex = np.argmax(np.abs(EquivalentChannels))
        return bestIndex

    def get_combined_channel(self, beam_index, channel_h):
        combined_channel = np.dot(np.squeeze(
            np.asarray(channel_h)), self.codebook[:,beam_index])
        combined_channel = float(np.abs(combined_channel))
        return combined_channel

# used


def arrayFactorGivenAngleForULA(self, numAntennaElements, theta, normalizedAntDistance=0.5, angleWithArrayNormal=0):
    indices = np.arange(numAntennaElements)
    if (angleWithArrayNormal == 1):
        arrayFactor = np.exp(
            1j * 2 * np.pi * normalizedAntDistance * indices * np.sin(theta))
    else:  # default
        arrayFactor = np.exp(
            1j * 2 * np.pi * normalizedAntDistance * indices * np.cos(theta))
    arrayFactor = arrayFactor / np.sqrt(numAntennaElements)
    return arrayFactor  # normalize to have unitary norm


def calc_omega(elevationAngles, azimuthAngles, normalizedAntDistance=0.5):
    sinElevations = np.sin(elevationAngles)
    omegax = 2 * np.pi * normalizedAntDistance * \
        sinElevations * np.cos(azimuthAngles)
    omegay = 2 * np.pi * normalizedAntDistance * \
        sinElevations * np.sin(azimuthAngles)
    return np.matrix((omegax, omegay))


def calc_vec_i(i, omega, antenna_range):
    print('a ', omega[:, i])
    print('b ', omega[:, i].shape)
    vec = np.exp(1j * omega[:, i] * antenna_range)
    print('c ', np.matrix(np.kron(vec[1], vec[0])).shape)
    return np.matrix(np.kron(vec[1], vec[0]))


def dft_codebook(dim):
    seq = np.matrix(np.arange(dim))
    mat = seq.conj().T * seq
    w = np.exp(-1j * 2 * np.pi * mat / dim)
    return w


def getNarrowBandULAMIMOChannel(azimuths_tx, azimuths_rx, p_gainsdB, number_Tx_antennas, number_Rx_antennas,
                                normalizedAntDistance=0.5, angleWithArrayNormal=0, pathPhases=None):
    """
    - assumes one beam per antenna element

    the first column will be the elevation angle, and the second column is the azimuth angle correspondingly.
    p_gain will be a matrix size of (L, 1)
    departure angle/arrival angle will be a matrix as size of (L, 2), where L is the number of paths

    t1 will be a matrix of size (nt, nr), each
    element of index (i,j) will be the received
    power with the i-th precoder and the j-th
    combiner in the departing and arrival codebooks
    respectively

    :param departure_angles: ((elevation angle, azimuth angle),) (L, 2) where L is the number of paths
    :param arrival_angles: ((elevation angle, azimuth angle),) (L, 2) where L is the number of paths
    :param p_gaindB: path gain (L, 1) in dB where L is the number of paths
    :param number_Rx_antennas, number_Tx_antennas: number of antennas at Rx and Tx, respectively
    :param pathPhases: in degrees, same dimension as p_gaindB
    :return:
    """
    azimuths_tx = np.deg2rad(azimuths_tx)
    azimuths_rx = np.deg2rad(azimuths_rx)
    # nt = number_Rx_antennas * number_Tx_antennas #np.power(antenna_number, 2)
    m = np.shape(azimuths_tx)[0]  # number of rays
    H = np.matrix(np.zeros((number_Rx_antennas, number_Tx_antennas)))

    gain_dB = p_gainsdB
    path_gain = np.power(10, gain_dB / 10)
    path_gain = np.sqrt(path_gain)

    # generate uniformly distributed random phase in radians
    if pathPhases is None:
        pathPhases = 2*np.pi * np.random.rand(len(path_gain))
    else:
        # convert from degrees to radians
        pathPhases = np.deg2rad(pathPhases)

    # include phase information, converting gains in complex-values
    path_complexGains = path_gain * np.exp(-1j * pathPhases)

    # recall that in the narrowband case, the time-domain H is the same as the
    # frequency-domain H
    for i in range(m):
        # at and ar are row vectors (using Python's matrix)
        at = np.matrix(arrayFactorGivenAngleForULA(number_Tx_antennas, azimuths_tx[i], normalizedAntDistance,
                                                   angleWithArrayNormal))
        ar = np.matrix(arrayFactorGivenAngleForULA(number_Rx_antennas, azimuths_rx[i], normalizedAntDistance,
                                                   angleWithArrayNormal))
        # outer product of ar Hermitian and at
        H = H + path_complexGains[i] * ar.conj().T * at
    factor = (np.linalg.norm(path_complexGains) / np.sum(path_complexGains)) * np.sqrt(
        number_Rx_antennas * number_Tx_antennas)  # scale channel matrix
    H *= factor  # normalize for compatibility with Anum's Matlab code

    return H


def arrayFactorGivenAngleForULA(numAntennaElements, theta, normalizedAntDistance=0.5, angleWithArrayNormal=0):
    '''
    Calculate array factor for ULA for angle theta. If angleWithArrayNormal=0
    (default),the angle is between the input signal and the array axis. In
    this case when theta=0, the signal direction is parallel to the array
    axis and there is no energy. The maximum values are for directions 90
        and -90 degrees, which are orthogonal with array axis.
    If angleWithArrayNormal=1, angle is with the array normal, which uses
    sine instead of cosine. In this case, the maxima are for
        thetas = 0 and 180 degrees.
    References:
    http://www.waves.utoronto.ca/prof/svhum/ece422/notes/15-arrays2.pdf
    Book by Balanis, book by Tse.
    '''
    indices = np.arange(numAntennaElements)
    if (angleWithArrayNormal == 1):
        arrayFactor = np.exp(
            1j * 2 * np.pi * normalizedAntDistance * indices * np.sin(theta))
    else:  # default
        arrayFactor = np.exp(
            1j * 2 * np.pi * normalizedAntDistance * indices * np.cos(theta))
    arrayFactor = arrayFactor / np.sqrt(numAntennaElements)
    return arrayFactor  # normalize to have unitary norm


def calc_rx_power(departure_angle, arrival_angle, p_gain, antenna_number, frequency=6e10):
    """This .m file uses a m*m SQUARE UPA, so the antenna number at TX, RX will be antenna_number^2.

    - antenna_number^2 number of element arrays in TX, same in RX
    - assumes one beam per antenna element

    the first column will be the elevation angle, and the second column is the azimuth angle correspondingly.
    p_gain will be a matrix size of (L, 1)
    departure angle/arrival angle will be a matrix as size of (L, 2), where L is the number of paths

    t1 will be a matrix of size (nt, nr), each
    element of index (i,j) will be the received
    power with the i-th precoder and the j-th
    combiner in the departing and arrival codebooks
    respectively

    :param departure_angle: ((elevation angle, azimuth angle),) (L, 2) where L is the number of paths
    :param arrival_angle: ((elevation angle, azimuth angle),) (L, 2) where L is the number of paths
    :param p_gain: path gain (L, 1) where L is the number of paths
    :param antenna_number: antenna number at TX, RX is antenna_number**2
    :param frequency: default
    :return:
    """
    departure_angle = np.deg2rad(departure_angle)
    arrival_angle = np.deg2rad(arrival_angle)
    c = 3e8
    mlambda = c / frequency
    k = 2 * np.pi / mlambda
    d = mlambda / 2
    nt = np.power(antenna_number, 2)
    m = np.shape(departure_angle)[0]
    nr = nt
    wt = dft_codebook(nt)
    wr = dft_codebook(nr)
    H = np.matrix(np.zeros((nt, nr)))

    # TO DO: need to generate random phase and convert gains in complex-values
    gain_dB = p_gain
    path_gain = np.power(10, gain_dB / 10)
    antenna_range = np.arange(antenna_number)

    def calc_omega(angle):
        sin = np.sin(angle)
        omegay = k * d * sin[:, 1] * sin[:, 0]
        omegax = k * d * sin[:, 0] * np.cos(angle[:, 1])
        return np.matrix((omegax, omegay))

    departure_omega = calc_omega(departure_angle)
    arrival_omega = calc_omega(arrival_angle)

    def calc_vec_i(i, omega, antenna_range):
        vec = np.exp(1j * omega[:, i] * antenna_range)
        return np.matrix(np.kron(vec[1], vec[0]))

    for i in range(m):
        departure_vec = calc_vec_i(i, departure_omega, antenna_range)
        arrival_vec = calc_vec_i(i, arrival_omega, antenna_range)
        H = H + path_gain[i] * departure_vec.conj().T * arrival_vec
    t1 = wt.conj().T * H * wr
    return t1


def getDFTOperatedChannel(H, number_Tx_antennas, number_Rx_antennas):
    wt = dft_codebook(number_Tx_antennas)
    wr = dft_codebook(number_Rx_antennas)
    dictionaryOperatedChannel = wr.conj().T * H * wt
    # dictionaryOperatedChannel2 = wr.T * H * wt.conj()
    # return equivalent channel after precoding and combining
    return dictionaryOperatedChannel


def getCodebookOperatedChannel(H, Wt, Wr):
    if Wr is None:  # only 1 antenna at Rx, and Wr was passed as None
        return H * Wt
    if Wt is None:  # only 1 antenna at Tx
        return Wr.conj().T * H
    return Wr.conj().T * H * Wt  # return equivalent channel after precoding and combining


def friis_propagation(Ptx, R, freq, gain=5):
    h = sc.c / freq
    # gain and effectiver aperture default for isotropic ideal antenna
    Prx = (20*np.log10((h/(4*np.pi*R)**2))+Ptx+gain)
    return Pr


def processChannelRandomGeo(data, spread=1, Nr=1, Nt=64):
    seed = 75648
    np.random.seed(seed)
    numRays = 2
    freq = 5e9
    Ptx = 30  # (db)
    angle_spread = spread
    departure = data[0]
    arrival = data[1]
    distance = data[2]

    gain = np.random.randn(numRays+10) + np.random.randn(numRays)
    #gain_in_dB = 20*np.log10(np.abs(gain))
    gain_in_dB = friis_propagation(Ptx, distance, freq, gain=gain)
    AoD_az = departure + angle_spread*np.random.randn(numRays)
    AoA_az = arrival + angle_spread*np.random.randn(numRays)
    phase = np.angle(gain)*180/np.pi

    Ht = getNarrowBandULAMIMOChannel(
        AoD_az, AoA_az, gain_in_dB, Nt, Nr, pathPhases=phase)
    Ht = Ht / np.linalg.norm(Ht)  # normalize channel to unit norm

    return Ht


def processCrazyChannels(angle, spread=1, Nr=1, Nt=64):
    seed = 75648
    np.random.seed(seed)
    numRays = 2
    #Ht = np.zeros((Nr,Nt))
    angle_spread = spread
    departure = angle
    if departure > 0:
        arrival = 180 + departure
    else:
        arrival = 180-departure

    gain = np.random.randn(numRays) + np.random.randn(numRays)
    gain_in_dB = 20*np.log10(np.abs(gain))
    AoD_az = departure + angle_spread*np.random.randn(numRays)
    AoA_az = arrival + angle_spread*np.random.randn(numRays)
    phase = np.angle(gain)*180/np.pi

    Ht = getNarrowBandULAMIMOChannel(
        AoD_az, AoA_az, gain_in_dB, Nt, Nr, pathPhases=phase)

    Ht = Ht / np.linalg.norm(Ht)  # normalize channel to unit norm

    return Ht


def processChannelRandomGeo_forNLOS(spread=1, Nr=1, Nt=64):
    seed = 75648
    np.random.seed(seed)
    angles = [35+90, 60]  # NLOS input range for ITU paper ->[20, 50]
    numRays = 2
    angle_spread = spread
    departure = angles[0]
    arrival = angles[1]

    #numValidChannels = 0
    #print('Processing ...')
    gain = np.random.randn(numRays) + np.random.randn(numRays)
    gain_in_dB = 20*np.log10(np.abs(gain))
    AoD_az = departure + angle_spread*np.random.randn(numRays)
    AoA_az = arrival + angle_spread*np.random.randn(numRays)
    phase = np.angle(gain)*180/np.pi

    Ht = getNarrowBandULAMIMOChannel(
        AoD_az, AoA_az, gain_in_dB, Nt, Nr, pathPhases=phase)

    Ht = Ht / np.linalg.norm(Ht)  # normalize channel to unit norm

    return Ht


if __name__ == '__main__':
    analogBeamformer = AnalogBeamformer()
    print('# codevectors=', analogBeamformer.get_num_codevectors())
    beam_index = 5
    channel_h = np.ones((1,32))
    print('gain mag=',analogBeamformer.get_combined_channel(beam_index, channel_h))
