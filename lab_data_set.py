#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lab seismology extension for the python implementation of the Adaptable Seismic Data
Format (ASDF).

TODO:
    add copyright and license?
"""

# Import base module (TODO: whole thing or just ASDFDataSet? Do I want access to errors?)
import pyasdf

# Import ObsPy to this namespace as well for methods and precision
import obspy

obspy.UTCDateTime.DEFAULT_PRECISION = 9

# Import plotting libraries
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np

# Imports for picking
import peakutils

# Add paths to my other modules
import sys
import os

_home = os.path.expanduser("~") + "/"
for d in os.listdir(_home + "git_repos"):
    sys.path.append(_home + "git_repos/" + d)


######### personal helpers ##########
# Specific to my data flow, maybe separate out to a different module


def create_stream_tpc5(tpc5_path: str, ord_stns: list, net="L0", chan="XHZ", hz=40e6):
    """
    Create an obspy Stream from a .tpc5 data file.
    ord_stns must be the list of stations in the order they appear in the tpc5 file.
    """
    import h5py
    import tpc5

    def get_stats(name, net="L0", chan="XHZ", hz=40e6):
        statn_stats = obspy.core.Stats()
        statn_stats.network = net
        statn_stats.channel = chan
        statn_stats.location = "00"
        statn_stats.sampling_rate = hz
        statn_stats.station = name
        return statn_stats

    f = h5py.File(tpc5_path, "r")

    # we need to know how many blocks to read
    # all channels have the same number of blocks, use channel 1
    chan_grp = f[tpc5.getChannelGroupName(1)]
    nblocks = len(chan_grp["blocks"].keys())
    ntr = len(f["measurements"]["00000001"]["channels"])
    source_stream = obspy.Stream()

    # TODO: stop trusting the save order, implement some explicit A1:AE05 map (tpc5-sxml)
    # iterate through stations, in whatever order they were saved
    # input saved channels as chan_nums because the tpc5 has no info about which channels were saved
    # tpc5 channels will always start at 1 and increase monotonically
    for tr in range(ntr):
        statn_stats = get_stats(ord_stns[tr], net=net, chan=chan, hz=hz)

        # iterate through continuous data segments
        # TranAX calls these Blocks, obspy calls them Traces
        for blk in range(1, nblocks + 1):
            # get the trace start time
            statn_stats.starttime = (
                obspy.UTCDateTime(
                    tpc5.getStartTime(f, 1)
                )  # gives the start of the whole recording
                + tpc5.getTriggerTime(f, 1, block=blk)  # seconds from start to trigger
                - tpc5.getTriggerSample(f, 1, block=blk)
                / statn_stats.sampling_rate  # seconds from trigger to block start
            )

            # get the raw voltage data
            raw = tpc5.getVoltageData(f, tr + 1, block=blk)
            # give the stats the length, otherwise it takes 0 points
            statn_stats.npts = len(raw)
            source_stream += obspy.Trace(raw, statn_stats)
    return source_stream


def setup_experiment_from_dir(exp_name: str, glob_str="*.tpc5"):
    """Auto setup new ASDF file based on files in this directory.
    exp_name :: str name of experiment, will create exp_name.h5 file
    glob_str :: limit data files read in by glob"""
    import glob

    # initialize dataset file
    ds = LabDataSet(exp_name + ".h5", compression="gzip-3")
    # find and add stations
    statxml_fp = glob.glob("*stations.xml")
    if not len(statxml_fp):
        raise Exception("No station xml found matching pattern *stations.xml")
    elif len(statxml_fp) > 1:
        raise Warning(
            "Warning: more than one station xml found! using {}".format(statxml_fp[0])
        )
    ds.add_local_locations(statxml_fp[0])
    # stat_locs and stns properties produced automatically now
    # find and add waveforms from tpc5 files
    wf_files = glob.glob(glob_str)  # doesn't add full path prefix
    ds.all_tags = []
    print("Adding waveforms from: ")
    for wf in wf_files:
        print(wf)
        tag = wf[:-5].lower()
        ds.add_waveforms(create_stream_tpc5(wf, ds.stns), tag)
    return ds


class LabDataSet(pyasdf.ASDFDataSet):
    """
    Object handling special Lab ASDF files and operations.
    """

    # don't override __init__

    def add_local_locations(self, statxml_filepath):
        """
        Add stations from a StationXML inventory which must have local locations as
        'extra' info. TODO: add statxml creator and reference here
        """
        inv = obspy.read_inventory(statxml_filepath, format="stationxml")
        nsta = len(inv[0].stations)
        stat_locs = {}
        stats_order = []  # retain A1-D4 order with AExx station codes
        for sta in inv[0].stations:
            sta_code = sta.code
            stats_order.append(sta_code)
            sta_loc = (
                [
                    float(sta.extra.local_location.value[xyz].value)
                    for xyz in ["x", "y", "z"]
                ]
                if hasattr(sta, "extra")
                else (np.NaN, np.Nan, np.Nan)
            )
            stat_locs[sta_code] = sta_loc

        # add the local_locations as a dictionary to never worry about shuffling the stations and locations
        # data can't take a dictionary (and requires a shape), but parameters takes the dictionary just fine
        self.add_auxiliary_data(
            data=np.array(stats_order, dtype="S"),
            data_type="LabStationInfo",
            path="local_locations",
            parameters=stat_locs,
        )

    # add property to quickly access station locations
    @property
    def stat_locs(self) -> dict:
        return self.auxiliary_data.LabStationInfo.local_locations.parameters

    @property
    def stns(self) -> list:
        """List of stations in the order of the stationxml file."""
        return list(
            np.char.decode(self.auxiliary_data.LabStationInfo.local_locations.data[:])
        )

    ######## picking methods of object ########
    def add_picks(self, tag, trace_num, picks):
        """Add a dict of picks for a (tag,trcnum) path.
        Picks in the form {stn:[picks]}
        Returns any overwritten picks as a safety."""
        # check for old_picks to overwrite
        try:
            old_picks = self.auxiliary_data.LabPicks[tag][f"tr{trace_num}"].parameters
            del self.auxiliary_data.LabPicks[tag][f"tr{trace_num}"]
        except:
            old_picks = {}
        self.add_auxiliary_data(
            data=np.array([]),
            data_type="LabPicks",
            path=f"{tag}/tr{trace_num}",
            parameters=picks,
        )
        return old_picks

    def plot_picks(
        self, tag, trace_num, view_from, view_len, new_picks, figname="picks_plot"
    ):
        """Produce an interactive plot of traces with numbered picks, and existing picks if present.
        Assumes 16 sensors.
        TODO: old_picks markers are too big"""
        fig, plotkey = subplts(4, 4)

        # are there existing picks to view_len?
        try:
            old_picks = self.auxiliary_data.LabPicks[tag][f"tr{trace_num}"].parameters
            plot_op = 1
        except:
            plot_op = 0

        for i, stn in enumerate(self.stns):
            # plot trace
            trc = self.waveforms["L0_" + stn][tag][trace_num].data
            fig.append_trace(
                go.Scattergl(
                    y=trc[view_from : view_from + view_len], line={"color": "black"}
                ),
                int(plotkey[i][0]),
                plotkey[i][1],
            )
            # plot existing picks, if any in window
            if plot_op and old_picks[stn][0] > view_from:
                fig.append_trace(
                    go.Scatter(
                        x=np.array(old_picks[stn]) - view_from,
                        y=trc[old_picks[stn]],
                        mode="markers",
                        marker={"symbol": "x", "color": "blue", "size": 10},
                    ),
                    int(plotkey[i][0]),
                    plotkey[i][1],
                )
            # plot new picks, if any in window
            if new_picks[stn][0] > view_from:
                fig.append_trace(
                    go.Scatter(
                        x=np.array(new_picks[stn]) - view_from,
                        y=trc[new_picks[stn]],
                        mode="markers+text",
                        text=[str(np) for np in range(len(new_picks[stn]))],
                        textposition="bottom center",
                    ),
                    int(plotkey[i][0]),
                    plotkey[i][1],
                )
        # plot the figure
        fig["layout"].update(showlegend=False)
        fig.write_html(figname + ".html")
        print(f"Picks plot written to {figname}.html")

    def interactive_check_picks(
        self, tag, trace_num, picks=None, view_from=180000, view_len=40000
    ):
        """Plot picks for all stations for one (tag,trcnum) and accept user adjustments.
        Auto-picks if no picks are provided or already stored.
        TODO: my notes imply that plotly is now interactive enough that I could replot after each input"""
        stns = self.stns  # TODO: is this necessary?

        # auto-pick if necessary
        try:
            old_picks = self.auxiliary_data.LabPicks[tag][f"tr{trace_num}"].parameters
        except:
            old_picks = {}
        if not picks:
            picks = {}
            if not old_picks:
                for stn in stns:
                    trc = self.waveforms["L0_" + stn][tag][trace_num].data
                    picks[stn] = auto_pick_by_noise(trc)
            else:
                picks = old_picks
                # TODO: there has to be better logic for this (still?)
        self.plot_picks(tag, trace_num, view_from, view_len, picks)

        # ask for inputs
        print(
            "Adjustment actions available: s for select pick, r for repick near, m for"
            " manual pick"
        )
        print("Enter as [chan][action key][index], e.g. s0 to select pick 0")
        adjust = input("Adjust a channel? - to exit: ")
        # TODO: make better multipick options
        while adjust != "-":
            # get channel
            try:
                chan = int(adjust[:2])
                action = adjust[2]
                num = int(adjust[3:])
            except:
                chan = int(adjust[0])
                action = adjust[1]
                num = int(adjust[2:])
            # parse action
            if action == "s":
                # select one correct pick
                picks[stns[chan]] = [picks[stns[chan]][num]]
            elif action == "r":
                # pick near somewhere else
                trc = self.waveforms["L0_" + stns[chan]][tag][trace_num].data
                picks[stns[chan]] = pick_near(
                    trc, num + view_from, reach_left=2000, reach_right=2000, thr=0.9
                )
            elif action == "m":
                # manually enter pick
                if num == -1:
                    picks[stns[chan]] = [-1]
                else:
                    picks[stns[chan]] = [num + view_from]
            # move to next adjustment or exit
            adjust = input("Adjust a channel? - to exit: ")

        # add picks, catching and returning overwritten old_picks
        old_picks = self.add_picks(tag, trace_num, picks)
        return old_picks

    ######## source location on object ########
    def locate_tag(self, tag, vp=0.272, bootstrap=False):
        """Locates all picked traces within a tag. vp in cm/s
        Assumes one pick per trace. TODO: fix that"""
        import scipy.optimize as opt

        stns = self.stns

        def curve_func_cm(X, a, b, c):
            t = np.sqrt((X[0] - a) ** 2 + (X[1] - b) ** 2 + 3.85 ** 2) / vp - c
            return t

        ntrcs = len(self.waveforms["L0_" + stns[0]][tag])
        for trace_num in range(ntrcs):
            # are there picks?
            try:
                picks = self.auxiliary_data.LabPicks[tag][f"tr{trace_num}"].parameters
            except:
                continue

            # associate locations
            xys = [self.stat_locs[stn][:2] for stn in stns if picks[stn][0] > 0]
            # picks to times
            arrivals = [picks[stn][0] / 40 for stn in stns if picks[stn][0] > 0]
            # sort xys and arrivals by arrival
            xys, arrivals = list(zip(*sorted(zip(xys, arrivals), key=lambda xa: xa[1])))
            # run the nonlinear least squares
            model, cov = opt.curve_fit(
                curve_func_cm,
                np.array(xys).T,
                np.array(arrivals) - arrivals[0],
                bounds=(0, [500, 500, 50]),
            )
            o_ind = [int((arrivals[0] - model[-1]) * 40)]
            self.add_auxiliary_data(
                data=model,
                data_type="Origins",
                path="{}/tr{}".format(tag, trace_num),
                parameters={"o_ind": o_ind, "cov": cov},
            )

    ######## content check ########
    def check_auxdata(self):
        """Report on presence of LabPicks and Origins, return True if both present.
        Prints if any tags are missing from either but doesn't affect output.
        No check on traces within tags.
        TODO: remove repetition, extend to other checks
        """
        tags = sorted(self.waveform_tags)
        try:
            lp_tags = self.auxiliary_data.LabPicks.list()
            if len(lp_tags) < len(tags):
                print(
                    "Picks missing for tags: "
                    + str([t for t in tags if t not in lp_tags])
                )
        except:
            print("No LabPicks!")
            return False
        try:
            loc_tags = self.auxiliary_data.Origins.list()
            if len(loc_tags) < len(tags):
                print(
                    "Origins missing for tags: "
                    + str([t for t in tags if t not in loc_tags])
                )
        except:
            print("No Origins!")
            return False
        print("LabPicks and Origins present for all other tags.")
        return True

    ######## get traces ########
    def get_traces(self, tag, trace_num, pre=200, tot_len=2048):
        """Return a dict of short traces from a tag/trcnum based on picks."""
        traces = {}
        picks = self.get_picks(tag, trace_num)
        for stn, pp in picks.items():
            if "L" in stn[:1]:
                stn = stn[3:]  # deal with existing picks having dumb station names
            pp = pp[0]
            sl = slice(pp - pre, pp - pre + tot_len)
            traces[stn] = self.waveforms["L0_" + stn][tag][trace_num].data[sl]
        return traces

    def get_picks(self, tag, trace_num):
        """Shortcut to return the picks dictionary."""
        return self.auxiliary_data.LabPicks[tag][f"tr{trace_num}"].parameters


######## picking helpers ########


def pick_near(trace, near, reach_left=2000, reach_right=2000, thr=0.9, AIC=[]):
    """Get a list of picks for one trace."""
    if len(AIC) == 0:
        AIC = get_AIC(trace, near, reach_left, reach_right)
    picks = (
        peakutils.indexes(AIC[5:-5] * (-1), thres=thr, min_dist=50) + 5
    )  # thres works better without zeros
    picks = list(picks + (near - reach_left))  # window indices -> trace indices
    # TODO: Do I need to remove duplicates and sort?
    return picks


def get_AIC(trace, near, reach_left=2000, reach_right=2000):
    """Calculate the AIC function used for picking. Accepts an Obspy trace or array-like data."""
    if hasattr(trace, "data"):
        window = trace.data[near - reach_left : near + reach_right]
    else:
        window = trace[near - reach_left : near + reach_right]
    AIC = np.zeros_like(window)
    for i in range(5, len(window) - 5):  # ends of window generate nan and -inf values
        AIC[i] = i * np.log10(np.var(window[:i])) + (len(window) - i) * np.log10(
            np.var(window[i:])
        )
    return AIC


def cut_at_rail(trace):
    """Find index where a signal rails."""
    dd = np.diff(trace)
    count = []
    for i, x in enumerate(dd):
        if x == 0:
            if len(count) == 0 or count[-1] == i - 1:  # first or next in a sequence
                count.append(i)
            else:  # new sequence
                count = [i]
            if len(count) > 9:
                break
    return count[0]


def auto_pick_by_noise(
    trace, noise_trig=10, noise_cut=50000, rl=3000, rr=3000, thresh=0.9
):
    """ Attempt to pick a trace without interaction, around the first point greater than noise_trig*noise_std."""
    cut = cut_at_rail(trace)
    trace = trace[:cut]
    # auto-aim for pick_near
    noise_std = np.std(trace[:noise_cut])
    near = np.argmax(trace > noise_std * noise_trig)  # argmax instead of argwhere()[0]
    try:
        picks = pick_near(trace, near, reach_left=rl, reach_right=rr, thr=thresh)
    except:
        picks = [-1]
    return picks


def subplts(row, col, titles="default"):
    if titles == "default":
        titles = ("chan {}".format(i) for i in range(row * col))
    fig = make_subplots(row, col, print_grid=False, subplot_titles=tuple(titles))
    plotkey = list(
        zip(
            np.hstack([[i + 1] * col for i in range(row)]),
            [i + 1 for i in range(col)] * col,
        )
    )
    return fig, plotkey


######## source helpers ########
def ball_force(
    diam=(1.18e-3), rho=7850, nu=[0.28, 0.3], E=[200e9, 6e9], h=0.305, Fs=40e6
):
    """
    calculate the force function from a ball drop
    radius in m, ball density in kg/m^3, ball and surface PR, ball and surface YM in Pa
    drop height in m, sampling freq. in Hz
    pmma defaults: 1190, .3-.34, 6.2e9
    steel: 7850, .28, 214e9"""
    radius = diam / 2
    v = np.sqrt(2 * 9.8 * h)
    d = sum((1 - np.array(nu)) / (np.pi * np.array(E)))
    tc = 4.53 * (4 * rho * np.pi * d / 3) ** 0.4 * radius * v ** -0.2
    fmax = 1.917 * rho ** 0.6 * d ** -0.4 * radius ** 2 * v ** 1.2
    ftime = np.arange(0, tc * 1.01, 1 / Fs)
    ffunc = -1 * np.nan_to_num(
        fmax * np.power(np.sin(np.pi * ftime / tc), 1.5)
    )  # times past tc are nan->0
    return tc, ffunc
