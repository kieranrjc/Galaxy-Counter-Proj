#v6

#--changelogs--
#v1 - implement atpy file reading, and ckdtree for aperture method, density plot, contour, heatmap
#v2 - file output fixed (for now), testing with masks 
#v3 - 'subset' automation w/out masks, vincent's distance for aperture, some housekeeping, some variables defined out of loop etc.
#v4 - Mask based subset automation (0.5 width redshift sets, and high/inter/low mass sets), boundary correction from a mask (in testing), per-galaxy slicing, generalised slice sizing,
#v5 - Nth Nearest Neighbour implementation w/ poisson boundary condition.
#v6 - Average colour sampling, new position array definition system, and table ID array.

# to do
# - adjust plotting for all subsets
# - find a way to calculate red fraction & overdensity (requires boundary correction) to produce Peng 2010 plot.
# - MAYBE work on 'Friends of Friends' implementation, 
# - continuation of Nth Nearest Neighbour.
# - optimise the core loop, vectorization etc.

# notes
# - construct tables using Table.where instead of having masked values, and add to table sets, much more verbose than 
#   exporting a new file for each slice.
# - exporting table sets isn't the same as topcat subsets, not sure how to change 'labels'
# - have to edit outdated atpy code to write to files because of deprecated astropy function. 
# - can't run code on uni PCs; no atpy, no perms, dl'd atpy doesn't work, no access to pip.
# - TOPCAT uses the inverse definition of numpy masked arrays to define subsets. np.ma.masked_inside() to define a slice for subsetting. Add mask as a new column and export. 
# - Limiting of nearest neighbour searches requires an upper distance, currently set at 10Mpc

#%%
import numpy as np, atpy, matplotlib.pyplot as plt, scipy.spatial as spy,time
from astropy.coordinates import Distance, ICRS
from astropy import units as u
from astropy.io import fits
from numba import jit
import sys 
from multiprocessing import Process,Queue
#%%Functions

#moved to function, to keep code tidy and consistent without having to redefine similar parameters twice.

@jit
def PixelCount(x,y,R,mask):
    N=0
    xmin=int(np.floor(x-R))          #defining a box around the current point for indexing the mask image.
    xmax=int(np.ceil(x+R))           #Slightly larger than the aperture radius to avoid rounding errors in counting.
    ymin=int(np.floor(y-R))
    ymax=int(np.ceil(y+R))
    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0
        
    (splity,splitx) = np.where(mask[0].data[ymin:ymax,xmin:xmax]==1)  #Finding all pixels inside the indexed box on the mask (fill value of -99),
                                                                             #mask in (y,x) co-ords, with Y-column returned first in the output tuple.
    maskpos =np.array([splitx+xmin,splity+ymin]).T                           #Values returned are co-ordinates within the box (starting at 0) so addition of min to re-centre
    
    if len(maskpos)>0:
        masktree=spy.cKDTree(maskpos)                                      
        N = len(masktree.query_ball_point([x,y],R))   
    
    return N

def joiner(q2,q):
    while True:
        print('\nJoiner: Awaiting table\n')
        t=q2.get()
        tfull2=q.get()
        print('\nJoiner: Table Received: '+tfull2.table_name+'\n')        
        if tfull2 == 'end':
            print('\nJoiner: Shutting Down\n')
            break 
        else:
            print('\nJoiner: Writing columns to table\n')
            for i in enumerate(tfull2.names):
                t.add_column(str(i[1]),tfull2[tfull2.names[int(i[0])]])
                                
            print('\nJoiner: '+ tfull2.table_name +' added to table.\n')    
            q2.put(t)
            
            
 
def func(j,q,tfull):
    
    imagemask=fits.open('DR11.multiband-mask.final.binary.fits',memmap=True)
    
    Q1=np.zeros(len(tfull))
    Q12=np.zeros(len(tfull))
    Q2=np.zeros(len(tfull))
    Q3=np.zeros(len(tfull))
    Q4=np.zeros(len(tfull))
    r=np.zeros(len(tfull))
    A=np.zeros(len(tfull))            #pre-allocation
    pixelr=np.zeros(len(tfull))
    pixelA=np.zeros(len(tfull))
    x=np.zeros(len(tfull))
    y=np.zeros(len(tfull))
    NeighNo = np.zeros(len(tfull),dtype=int)
    avguv = np.zeros(len(tfull))
    avgub = np.zeros(len(tfull))
    avgvj = np.zeros(len(tfull))
    Aperuv = np.zeros(len(tfull))
    Aperub = np.zeros(len(tfull))
    Neighuv = np.zeros(len(tfull))
    Neighub = np.zeros(len(tfull))
    Neighvj = np.zeros(len(tfull))
    Apervj = np.zeros(len(tfull))
    Aperz = np.zeros(len(tfull))
    avgz = np.zeros(len(tfull))
    NNz = np.zeros(len(tfull))
    
    
    pixeldegr = 0.2684/3600         #pixel/degree ratio
    pixelsqdegr = pixeldegr**2
    slicesize=.5                    #width of redshift slice
    radius = [0.05,0.1,0.15,0.25,0.5,1]                      #Aperture Radius in Mpc 
    NN = [3,5,7,10,25,50]                          #Nearest Neighbour number
    x=tfull.X_IMAGE               #moved out of the loop to avoid constant reassignment
    y=tfull.Y_IMAGE
    zmax=tfull.z+(slicesize/2)
    zmin=tfull.z-(slicesize/2)

    print('\nStarting run '+str(j+1)+' \nAperture size - '+str(radius[j])+' Mpc\nNearest Neighbours - '+str(NN[j])+'\n ')
    
    imax = len(tfull)
    irange = range(imax)
    
    for i in irange:
            
        mask=np.ma.masked_outside(tfull.z,zmin[i],zmax[i]).mask     #redshift slice mask; only creates a N*1 mask, have to manually apply to other columns/
        ID = np.where(mask==False)[0] # table index numbers NOT tfull.ID
        RA = tfull.RA[ID]       #applying mask to other table variables for compiling into a list of positions
        DEC = tfull.DEC[ID]
        pos = np.array([RA,DEC]).T   #indexing using the inverse of the redshift mask. 
        
        tree = spy.cKDTree(pos)                                     #Position tree inside redshift slice
        avgz[i] = np.mean(np.ma.masked_array(tfull.z,mask=mask))
        
        if j==0:
            avguv[i] = np.mean(tfull.U[ID]-tfull.V[ID])
            avgub[i] = np.mean(tfull.U[ID]-tfull.B[ID])
            avgvj[i] = np.mean(tfull.V[ID]-tfull.J[ID])        
            
    
        c1 = ICRS(ra = 0*u.degree, dec=0*u.degree, distance=Distance(z = avgz[i])) #defining two coordinates 1 degree apart
        c2 = ICRS(ra = 1*u.degree, dec=0*u.degree, distance=Distance(z = avgz[i])) #
        onedegreesep = c1.separation_3d(c2)                                     #so 1 degree = onedegreesep                     
#        
        r[i]=radius[j]/onedegreesep.value                    
    
        pixelr[i]=r[i]/pixeldegr       
        
        (distances,Q11) = tree.query([tfull.RA[i],tfull.DEC[i]],k=NN[j]+1,distance_upper_bound=10*(r[i]/radius[j])) # upper bound to help mask definition
        
        distances=distances[np.isfinite(distances)]     #when NN exceeds number of neighbours, rest of table is filled with infs, recreates array using distance values
        Q11=Q11[:len(distances)]
                
        NeighNo[i] = int(len(distances)-1)                      #number of neighbours (useful for tracking where infs occur)
        
        Q1[i]=distances[-1]                             #-1 index selects final value (the distance we're interested in)
        
        Q12[i] = Q1[i]*onedegreesep.value                #distance in Mpc
    
        Q21 = tree.query_ball_point([tfull.RA[i],tfull.DEC[i]],r[i])
        
        Q2[i] = len(Q21)      #counts number of galaxies within aperture radius.
        
        Q3[i] = PixelCount(x[i],y[i],pixelr[i],imagemask)      
        
        NeighID = ID[Q11] #subsets of ID detected in neighbour searches
        AperID = ID[Q21]
    
        Aperuv[i]  = np.mean(tfull.U[AperID]-tfull.V[AperID])         #environment sample average colour
        Aperub[i]  = np.mean(tfull.U[AperID]-tfull.B[AperID])
        Apervj[i]  = np.mean(tfull.V[AperID]-tfull.J[AperID])
        Aperz[i]   = np.mean(tfull.z[AperID])
            
        
        if NeighNo[i] == NN[j] : #avoids 'unreliable' data that don't have the full number of neighbours. 
        
            NNz[i] = np.mean(tfull.z[NeighID])
            Neighuv[i] = np.mean(tfull.U[NeighID]-tfull.V[NeighID])
            Neighub[i] = np.mean(tfull.U[NeighID]-tfull.B[NeighID]) 
            Neighvj[i] = np.mean(tfull.V[NeighID]-tfull.J[NeighID])
            Q4[i] = PixelCount(x[i],y[i],Q1[i]/pixeldegr,imagemask)
        
        if (i*205.91)%imax ==0:                                                           #Progress checker using time.time()
            tcheck = time.time()
            telapsed = tcheck-tstart
            print('Run: '+str(j+1)+' - '+str(np.around(100*i/imax,decimals=2)) +'% Time Elapsed: '+str(int(telapsed//3600))+':' + str(int(telapsed % 3600//60))+':'+str(np.around(telapsed%3600%60,decimals=2)))
            
    tcheck = time.time()
    telapsed = tcheck-tstart
    print('\nRun '+str(j+1)+' Looping complete: '+str(int(telapsed//3600))+':' + str(int(telapsed % 3600//60))+':'+str(np.around(telapsed%3600%60,decimals=2)))
    
           
    A = np.pi*radius[j]**2   
    degA=np.pi*r**2             #Aperture and pixel areas for table calculation
    
    pixelA = degA/pixelsqdegr
    
    MaskPix = Q3/pixelA
    GoodPix = 1-MaskPix
    
    AperDens = Q2/A
    CorrectedAperPop = Q2/GoodPix
    CorrectedAperDens = (Q2/GoodPix)/A
    
    NNAdeg= np.pi*Q1**2
    NNAMpc= np.pi*Q12**2
    NNPixelA = NNAdeg/(pixelsqdegr)
    
    pixdensity1 = (NeighNo+1)/NNPixelA          #density1 represents the entire area spanned by the Nearest Neighbour Distance
    rate1 = Q4*pixdensity1             #interval defined from area density. 
    P1 = 1-np.exp(-rate1)          #probability of not 0 galaxies in a poisson distribution. 
    
    pixdensity2 = (NeighNo+1)/(NNPixelA-Q4)     #density 2 represents a reduced area, with the mask pixels removed, compairsons have to be made.
    rate2 = Q4*pixdensity2
    P2 = 1-np.exp(-rate2)
    
    Mpcdensity = (NeighNo+1)/NNAMpc
    tfull2=atpy.Table(name='Run '+str(j)+': '+ str(NN[j])+' NN - '+str(radius[j])+ ' Mpc Aperture' ) 
    print('\nRun '+str(j+1)+' Bulk Calculation Complete.\n')
    
    if j == 0:
        tfull2.add_column('Average_Slice_U-B',avgub)
        tfull2.add_column('Average_Slice_U-V',avguv)
        tfull2.add_column('Average_Slice_V-J',avgvj)           
        tfull2.add_column('Average_Slice_z',avgz)
        
    tfull2.add_column(str(radius[j])+'MPC_Aperture_Radius',r)  
    tfull2.add_column(str(radius[j])+'MPC_Aperture_Radius_Pixels',pixelr)
    tfull2.add_column(str(radius[j])+'MPC_Aperture_Area',A)
    tfull2.add_column(str(radius[j])+'MPC_Aperture_Pixels',pixelA)
    tfull2.add_column(str(radius[j])+'MPC_Aperture_Mask_Pixels',Q3)
    tfull2.add_column(str(radius[j])+'MPC_Aperture_Mask_Pixel_Fraction',MaskPix)
    tfull2.add_column(str(radius[j])+'MPC_Aperture_Good_Pixel_Fraction',GoodPix)
    tfull2.add_column(str(radius[j])+'MPC_Aperture_Radial_Population',Q2)
    tfull2.add_column(str(radius[j])+'MPC_Aperture_Number_Density_sq_Mpc',AperDens)
    tfull2.add_column(str(radius[j])+'MPC_Aperture_Corrected_Radial_Population',CorrectedAperPop)
    tfull2.add_column(str(radius[j])+'MPC_Aperture_Corrected_Number_Density_sq_Mpc',CorrectedAperDens)
    tfull2.add_column(str(radius[j])+'MPC_Aperture_Average_U-B',Aperub)  
    tfull2.add_column(str(radius[j])+'MPC_Aperture_Average_U-V',Aperuv)  
    tfull2.add_column(str(radius[j])+'MPC_Aperture_Average_V-J',Apervj)  
    tfull2.add_column(str(radius[j])+'MPC_Aperture_Average_redshift',Aperz)  
    
    tfull2.add_column(str(NN[j])+'NN_Neighbours',NeighNo)
    tfull2.add_column(str(NN[j])+'NN_Distance_Mpc',Q12)
    tfull2.add_column(str(NN[j])+'NN_Area_Mpc',NNAMpc)
    tfull2.add_column(str(NN[j])+'NN_Pixel_Area',NNPixelA)
    tfull2.add_column(str(NN[j])+'NN_Density_Pixels',pixdensity1)
    tfull2.add_column(str(NN[j])+'NN_Mask_Area_Rate',rate1)
    tfull2.add_column(str(NN[j])+'NN_P1',P1)
    tfull2.add_column(str(NN[j])+'NN_Reduced_Area_Density_Pixels',pixdensity2)
    tfull2.add_column(str(NN[j])+'NN_Reduced_Mask_Area_Rate',rate2)
    tfull2.add_column(str(NN[j])+'NN_P2',P2)
    tfull2.add_column(str(NN[j])+'NN_Density_Mpc',Mpcdensity)
    tfull2.add_column(str(NN[j])+'NN_Average_U-B',Neighub)
    tfull2.add_column(str(NN[j])+'NN_Average_U-V',Neighuv)
    tfull2.add_column(str(NN[j])+'NN_Average_V-J',Neighvj)
    tfull2.add_column(str(NN[j])+'NN_Average_redshift',NNz)
    
    for i in np.arange(0.2,1.2,0.2):                                             #probability subsets
        submask1 = np.ma.masked_less_equal(P1,i).mask
        submask2 = np.ma.masked_less_equal(P2,i).mask
        tfull2.add_column(str(NN[j])+'NN Galaxy in mask probability <'+str(int(i*100))+'%',submask1)
        tfull2.add_column(str(NN[j])+'NN Galaxy in mask reduced probability <'+str(int(i*100))+'%',submask2)
        
    submask = np.ma.masked_equal(NeighNo,NN[j]).mask
    tfull2.add_column(str(NN[j])+'NN_Candidates',submask)    
    print('\nRun '+str(j+1)+' Writing Data Columns Complete\n')
    
    q.put(tfull2)
    

tstart = time.time()
    
if __name__ == '__main__':
    var=[0,1,2,3,4,5]
    procs=[]
    confirm=input('Confirm running? This will result in the erasing of previously set variables. y/n \n')
    
    if confirm != 'y':
        sys.exit()
        
    tfull=atpy.Table('DR11-BestGal.fits')                                    #sample and mask import
    
    queue=Queue()
    q2=Queue()
    q2.put(tfull)

    for index,number in enumerate(var):
        proc = Process(target=func,args=(number,queue,tfull))
        procs.append(proc)
        proc.start()
        
    joinerproc= Process(target=joiner,args=(q2,queue))      
    joinerproc.start()
    
    for proc in procs:
        proc.join()
        
        
    print('\nJoiner end pushed\n')
    queue.put('end')
    tfinal = q2.get()
       
    
    print(str(len(var))+' variable runs complete')
    
    print('Creating Subset Masks')
    
    for i in np.arange(0,max(tfull.z),0.5):                           #creating redshift slice subsets
        submask = np.ma.masked_inside(tfull.z,i,i+0.5).mask
        tfinal.add_column('z '+ str(i) + ' to ' + str(i+0.5),submask)
    

        
    IMmask=np.ma.masked_inside(tfinal.mass,4e8,2e9).mask                #mass subsets
    HMmask=np.ma.masked_greater(tfinal.mass,2e9).mask
    LMmask=np.ma.masked_less(tfinal.mass,4e8).mask
    
    tfinal.add_column('High_mass',HMmask)                            
    tfinal.add_column('Low_mass',LMmask)
    tfinal.add_column('Intermediate_mass',IMmask)
    
    print('Masks added to table')
    
    tfinal.write('multiproc2testfinalhope.fits',overwrite=True)
    
    print('Writing Table to File Complete')    
    hjk=input('\nbreak\n')

