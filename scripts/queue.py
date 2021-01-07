import time
import shlex
import os,sys
import subprocess 

class WorkerBsubPsana:
    def __init__(self,command=None,max_run_time=3600*8):
        self.command = command
        self.start_time = time.time()
        self.subprocess_try = False
        self.subprocess_success = False 
        self.submit_out = None 
        self.submit_err = None 
        self.subprocess_err = None 
        self.__status__ = "waiting"
        self.max_run_time = max_run_time

    def start(self):
        try:
            self.subprocess_try = True
            self.p = subprocess.Popen(shlex.split(self.command),stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.subprocess_success = True
        except Exception as error:
            self.subprocess_err = error
            self.__status__ = "exit"
        try:
            self.submit_out,self.submit_err = self.p.communicate() 
            time.sleep(10) # let bjobs to refresh
        except:
            self.subprocess_success = False
            self.__status__ = "exit"

    @staticmethod
    def bjobs_output(jobid=None,jobname=None,channel=""):
        # channel = "", "-d", "-p", "-r"
        # return: out is string with \n, not a list
        if jobid is None and jobname is None:
            cmd = 'bjobs ' + channel
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            out, err = process.communicate()
            try: 
                process.kill()
            except: 
                pass
            process.wait() 
        elif jobid is not None and jobname is None:
            cmd = "bjobs " + channel + " " + str(jobid)
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            out, err = process.communicate()
            try: 
                process.kill()
            except: 
                pass
            process.wait() 
        elif jobid is None and jobname is not None:
            cmd = 'bjobs -J ' + '*\"' + jobname + '\"*' + ' ' + channel 
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            out, err = process.communicate()
            try: 
                process.kill()
            except: 
                pass
            process.wait() 
        elif jobid is not None and jobname is not None:
            cmd = 'bjobs -J ' + '*\"' + jobname + '\"*' + ' ' + channel + ' ' + str(jobid)
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            out, err = process.communicate()
            try: 
                process.kill()
            except: 
                pass
            process.wait()
        return out
    
    @staticmethod
    def kill_job_by_jobid(jobid):
        cmd = "bkill " + str(jobid)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        out,err = process.communicate()
        try: 
            process.kill()
        except: 
            pass
        process.wait()
    
    @staticmethod
    def force_kill_job_by_jobid(jobid):
        cmd = "bkill -r " + str(jobid)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        out,err = process.communicate()
        try: 
            process.kill()
        except: 
            pass
        process.wait()

    @staticmethod
    def jobname_to_jobids(jobname):
        jobids = []
        out = WorkerBsubPsana.bjobs_output(jobname=jobname)
        for s in out.split("\n"):
            if str.isdigit(s[:2]):
                jobids.append(int(s.split()[0]))
        out = WorkerBsubPsana.bjobs_output(jobname=jobname,channel="-d")
        for s in out.split("\n"):
            if str.isdigit(s[:2]):
                jobids.append(int(s.split()[0]))
        return jobids

    @staticmethod
    def get_status_by_jobname(jobname=None):
        """
        - pending,running,done,exit,suspended,nojob
        """
        if jobname is None:
            return "nojob"
        
        jobids = WorkerBsubPsana.jobname_to_jobids(jobname)
        if len(jobids)==0:
            return "nojob"
        
        status = []
        for jobid in jobids:
            out = WorkerBsubPsana.bjobs_output(jobid=jobid)
            if "susp" in out.lower():
                status.append("suspended")
            elif "pend" in out.lower():
                status.append("pending")
            elif "done" in out.lower():
                status.append("done")
            elif "exit" in out.lower():
                status.append("exit")
            elif "run" in out.lower():
                status.append("running")
            else:
                status.append("exit")
        
        if "exit" in status:
            return "exit"
        if "suspended" in status:
            return "suspended"
        if "pending" in status:
            return "pending"
        if "running" in status:
            return "running"
        return "done"
        

    @staticmethod
    def get_status_by_jobid(jobid=None): 
        """
        - pending,running,done,exit,suspended,nojob
        """
        if jobid is None:
            return "nojob"
            
        out = WorkerBsubPsana.bjobs_output(jobid=jobid) 
        if "susp" in out.lower():
            return "suspended"
        elif "pend" in out.lower():
            return "pending"
        elif "done" in out.lower():
            return "done"
        elif "exit" in out.lower():
            return "exit"
        elif "run" in out.lower():
            return "running"
        elif "found" in out.lower():
            return "nojob"
        else:
            return "exit"

        
    def status(self):
        if self.__status__ == "done":
            return "done"
        if self.__status__ == "exit":
            return "exit"
        if not self.subprocess_try:
            return "waiting"
        if not self.subprocess_success:
            return "exit"
        if self.p.poll() is None:
            return "running"
        elif self.p.poll() > 0:
            return "exit"
        elif self.p.poll() < 0:
            return "exit"
        
        if not hasattr(self,"jobid"):
            import re
            self.p.poll() 
            out,err = None,None
            try: 
                out, err = self.p.communicate()
            except: 
                pass
            self.submit_out = self.submit_out or out
            self.submit_err = self.submit_err or err
            jobid = re.findall("<(.*?)>",self.submit_out)
            if len(jobid)==0:
                self.jobid = None
            else:
                self.jobid = int(jobid[0])
        
        if self.jobid:
            self.__status__ = WorkerBsubPsana.get_status_by_jobid(self.jobid)
            return self.__status__
        return "nojob"

    def success(self):
        if self.status() in ["done"]:
            return True
        return False

    def wait(self):
        while True:
            if self.status() in ["waiting","done","exit","suspended","nojob"]:
                return self.status()
            
            if self.runtime()>self.max_run_time:
                self.terminate()
                return "exit"
            
            time.sleep(5)

    def terminate(self):
        self.status()
        try:
            WorkerBsubPsana.kill_job_by_jobid(self.jobid)
        except:
            pass 
        try: 
            self.p.terminate()
        except: 
            pass
        try: 
            self.p.kill()
        except: 
            pass
        try: 
            self.p.wait()
        except: 
            pass
 
    def runtime(self):
        return time.time() - self.start_time
        