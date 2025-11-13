module fltrace
    implicit none
    contains

    subroutine trace_fieldlines(seeds, r, s, p, br, bs, bp, step_size, max_steps, save_flag, nlines_out, &
    nseeds, image_res, image_extent, image_parameters, nr, ns, np, xl, emission_matrix)
    integer,parameter :: rk=selected_real_kind(15,100)
    integer, intent(in):: nseeds, nr, ns, np, max_steps, nlines_out, image_res
    real(rk), intent(in):: seeds(1:nseeds,1:3)
    real(rk), intent(in):: br(1:nr,1:ns+1,1:np+1), bs(1:nr+1,1:ns,1:np+1), bp(1:nr+1,1:ns+1,1:np)
    real(rk), intent(in):: r(1:nr), s(1:ns), p(1:np)
    real(rk), intent(in):: step_size
    real(rk), dimension(:,:,:,:,:), allocatable:: db
    logical, intent(in):: save_flag
    real(rk), intent(out), dimension(1:nlines_out,1:max_steps,1:3):: xl  !Array for the saved field lines, if necessary.
    real(rk):: dr, ds, dp
    real(rk), dimension(64,64) :: m

    ! --- Image generation stuff
    real(rk), intent(out), dimension(1:image_res,1:image_res):: emission_matrix !For plotting things in the plane of earth-observations
    real(rk), intent(in):: image_extent !For plotting things in the plane of earth-observations
    real(rk), dimension(1:100):: image_parameters

    allocate(db(8,nr,ns,np,3))
    db = 0.0_rk

    call prepareInterpB(br, bs, bp, db, m, r, s, p, nr, ns, np, dr, ds, dp)

    call find_fieldlines(seeds, xl, db, m, step_size, save_flag, r, s, p, nr, ns, np, dr, ds, dp, image_res, image_extent, image_parameters, emission_matrix) !Integrates along the field lines using the existing tracer, saves data to the array xl

    end subroutine trace_fieldlines

    !****************************************************************
    subroutine find_fieldlines(x0, xl, db, m, maxdl, save_flag, r, s, p, nr, ns, np, dr, ds, dp, image_res, image_extent, image_parameters, emission_matrix)
    integer,parameter :: rk=selected_real_kind(15,100)
    real(rk), intent(in), dimension(:,:) :: x0
    logical, intent(in):: save_flag
    real(rk), intent(inout), dimension(:,:,:):: xl
    real(rk), dimension(:,:,:,:,:):: db
    real(rk), intent(in):: r(1:nr), s(1:ns), p(1:np)
    real(rk), intent(in):: dr, ds, dp
    integer, intent(in):: nr, ns, np
    real(rk), dimension(64,64) :: m
    ! ---
    integer :: nfl0, nmax, n, startn, endn, line_length
    real(rk) :: rMin, rMax, maxerror, minB
    real(rk) :: maxdl, dl, r1, r2, dl_rkt, error, k1r, rl
    real(rk), dimension(3) :: x1, x2, dx1, dx2, k1, k2, xstart, xend
    real(rk), dimension(:,:):: xl_local(1:size(xl,2),1:3)
    integer :: nxt, cntr, dirn, i

    real(rk):: surface_prop, open_prop, maxb_surface, maxb_overall, maxheight
    integer, dimension(:), allocatable :: endflag

    ! ---
    integer, intent(in):: image_res
    real(rk), intent(out), dimension(1:image_res,1:image_res):: emission_matrix
    real(rk), dimension(1:image_res,1:image_res):: local_matrix
    real(rk), dimension(1:100):: image_parameters
    real(rk), intent(in):: image_extent !For plotting things in the plane of earth-observations

    emission_matrix = 0.0_rk; local_matrix = 0.0_rk
    nfl0 = size(x0,1)   !Number of field lines to be traced. I assume xl is a list of the start points, but can't be entirely sure. Let's have a cup of coffee.
    nmax = size(xl,2)

    rMin = dexp(r(1)); rMax = dexp(r(nr))
    allocate(endflag(1:nfl0))
    ! Initialize all field lines to "incomplete" status:
    endflag = 0
    maxerror = 1.0_rk
    ! Main loop:
    print*, 'Tracing', nfl0, 'field lines'

    maxb_surface = maxval(dsqrt(sum(db(1,1,:,:,:)**2, 3)))
    maxb_overall = maxval(dsqrt(sum(db(1,:,:,:,:)**2, 3)))
    minB = 0.0000001_rk*maxb_overall
    xl = 0.0_rk
    !$omp parallel do default(shared) &
    !$omp& private(i, dirn, nxt, cntr, dl, dl_rkt, rl, k1r, r1, r2,  &
    !$omp& x1, x2, xstart, xl_local, xend, k1, k2, dx1, dx2, error, local_matrix,  &
    !$omp& startn, endn, line_length, maxheight, surface_prop, open_prop)
    do i = 1, nfl0
        if (nfl0 .le. 100000) then
            if ((mod(i,1000) == 0 .or. i == nfl0))            print*, 100*i/nfl0, '% complete'
        else
            if ((mod(i,nfl0/100) == 0 .or. i == nfl0))            print*, 100*i/nfl0, '% complete'
        end if

        ! Initialise variables:
        xl_local = 0.0_rk
        xl_local(:,3) = -99.0_rk !Flag to ensure the length of the fieldline is appropriately measured

        local_matrix = 0.0_rk
        maxheight = 1.0_rk

        do dirn=-1,1,2
            ! Reverse direction of field for backward and forward tracing:
            dl = maxdl

            nxt = nmax/2   !Start in the middle and shift things about later if necessary, that's quite clever actually.

            cntr = 0

            ! Add startpoint to output array:
            xl_local(nxt,:) = x0(i,:)
            ! Interpolate k1:
            call interpB(xl_local(nxt,:), k1, db, m, r, s, p, nr, ns, np, dr, ds, dp)

            dl_rkt = dsqrt(sum(k1*k1))

            if (dl_rkt < minB) exit   ! stop if null is reached
            dl_rkt = dl_rkt*dirn
            k1 = k1/dl_rkt
            do
                ! Compute midpoint:
                x2 = xl_local(nxt,:) + dl*k1
                r2 = dsqrt(sum(x2*x2))

                ! If outside boundary, do Euler step to boundary and stop:
                if ((r2 .lt. rMin).or.(r2 .gt. rMax)) then
                    rl = dsqrt(sum(xl_local(nxt,:)*xl_local(nxt,:)))
                    k1r = (xl_local(nxt,1)*k1(1) + xl_local(nxt,2)*k1(2) &
                        + xl_local(nxt,3)*k1(3))/rl
                    if (r2.lt.rMin) then
                        dl = (rMin - rl)/k1r
                    else
                        dl = (rMax - rl)/k1r
                    end if
                    nxt = nxt + dirn
                    cntr = cntr + 1
                    if (cntr .ge. nmax/2) then
                        print*,'This field line is very long. Aborting...'
                        print*, 'If this keeps happening, something more sinister is probably afoot.'
                        xl_local(:,:) = -99.0_rk
                        exit
                    end if

                    xl_local(nxt,:) = xl_local(nxt-dirn,:) + dl*k1

                    exit
                end if
                ! Interpolate k2:
                call interpB(x2, k2, db, m, r, s, p, nr, ns, np, dr, ds, dp)

                dl_rkt = dsqrt(sum(k2*k2))
                dl_rkt = dl_rkt*dirn

                ! Compute first and second-order update:
                k2 = 0.5_rk*(k1 + k2/dl_rkt)

                if (sum(k2*k2) < minB*minB) exit   ! stop if null is reached
                dx1 = dl*k1
                dx2 = dl*k2

                ! Estimate error from difference:
                error = sum((dx1-dx2)*(dx1-dx2))

                ! Modify step size depending on error:
                if (error.lt.maxerror) then
                    dl = maxdl
                else
                    dl = min(maxdl, 0.85_rk*dabs(dl)*(maxerror/error)**0.25_rk)
                end if

                ! Update if error is small enough:
                if (error.le.maxerror) then

                    x1 = xl_local(nxt,:) + dx2
                    r1 = dsqrt(sum(x1*x1))
                    ! Return midpoint if full step leaves domain:
                    if ((r1.lt.rMin).or.(r1.gt.rMax)) x1 = x2
                    nxt = nxt + dirn
                    cntr = cntr + 1
                    if (cntr .ge. nmax/2) then
                        print*,'This field line is very long. Aborting...'
                        print*, 'If this keeps happening, something more sinister is probably afoot.'
                        xl_local(:,:) = -99.0_rk
                        exit
                    end if
                    xl_local(nxt,:) = x1

                    ! Interpolate k1 at next point:
                    call interpB(x1, k1, db, m, r, s, p, nr, ns, np, dr, ds, dp)
                    dl_rkt = dsqrt(sum(k1**2))

                    if (image_res > 0) then
                        call update_emissions(local_matrix, x1, k1, r, nr, image_parameters, maxb_overall, image_extent)
                    end if

                    if (dsqrt(sum(x1**2)) > maxheight) then
                        maxheight = dsqrt(sum(x1**2))
                    end if

                    if (dl_rkt < minB) exit   ! stop if null is reached
                    dl_rkt = dl_rkt*dirn
                    k1 = k1/dl_rkt
                end if
            end do
          end do

        startn = 1; endn = 0
        do n = 1, nmax
            if (xl_local( n, 3) > -90_rk) then
                endn = n
            else if (endn < 1) then
                startn = n + 1
            end if
        end do
        !Shift such that the field line is at the start of the array, for ease of transferring the data

        if (startn < nmax .and. (endn - startn) > 5) then
            xl_local(1:endn-startn+2,:) = xl_local(startn:endn+1,:)
            xl_local(endn-startn+2:nmax,:) = 0.0_rk
            line_length = endn - startn + 1
            !Determine the weighting based on the magnetic field strength on the solar surface. Need to determine which of the ends meet the surface, for a start
            !If it is a closed field line, take the mean of both values
            xstart =  xl_local(1,:)
            xend = xl_local(line_length,:)
            if ((sum(xstart**2) < 1.1_rk*dexp(r(1))**2) .and. (sum(xend**2) < 1.1_rk*dexp(r(1))**2)) then !Closed field line
                call interpB(xstart, k1, db, m, r, s, p, nr, ns, np, dr, ds, dp)
                dl_rkt = dsqrt(sum(k1*k1))
                surface_prop = 0.5_rk*dl_rkt/maxb_surface
                call interpB(xend, k1, db, m, r, s, p, nr, ns, np, dr, ds, dp)
                dl_rkt = dsqrt(sum(k1*k1))
                surface_prop = surface_prop + 0.5_rk*dl_rkt/maxb_surface
            else if (sum(xstart**2) < 1.1_rk*dexp(r(1))**2) then
                call interpB(xstart, k1, db, m, r, s, p, nr, ns, np, dr, ds, dp)
                dl_rkt = dsqrt(sum(k1*k1))
                surface_prop = dl_rkt/maxb_surface
            else if (sum(xend**2) < 1.1_rk*dexp(r(1))**2) then
                call interpB(xend, k1, db, m, r, s, p, nr, ns, np, dr, ds, dp)
                dl_rkt = dsqrt(sum(k1*k1))
                surface_prop = dl_rkt/maxb_surface
            else !This field line never touches the surface, so don't plot it at all.
                xl_local(:,:) = 0.0_rk
                line_length = 0
                surface_prop = 0.0_rk
            end if
        else
            xl_local(:,:) = 0.0_rk
            line_length = 0
        end if

        !Determine weighting for openness. I think perhaps doing this as a proportion of the maximum height might be nice? Stops the massive arcades dominating.
        open_prop = maxheight/dexp(r(nr))

        if (line_length > 0 .and. image_res > 0) then
            emission_matrix = emission_matrix + (open_prop**image_parameters(3))*(surface_prop**abs(image_parameters(2)))*local_matrix
        end if

        if (save_flag) then
            xl(i,:,:) = xl_local(:,:)
        end if

    end do
    !$omp end parallel do

    end subroutine find_fieldlines

    !****************************************************************

    subroutine update_emissions(local_matrix, x1, k1, r, nr, image_parameters, maxb_overall, image_extent)
    integer,parameter :: rk=selected_real_kind(15,100)
    real(rk), dimension(:,:):: local_matrix
    real(rk), dimension(1:100):: image_parameters
    real(rk), dimension(3):: x1, k1
    real(rk):: thomson_angle, thomson_factor
    real(rk):: yfact, zfact, altitude, longitude, bmag, overall_factor, maxb_overall
    integer:: y_res, z_res, y_index, z_index
    real(rk), intent(in):: r(1:nr)
    integer, intent(in):: nr
    real(rk), intent(in):: image_extent !For plotting things in the plane of earth-observations

    !I've thought about the geometry a bit more, and I think a different approach is required. Bugger.
    !Determine appropriate point in the y, z, plane
    !Image dimensions are determined by the maximum radius (obvs)
    !local_matrix = 0.0_rk
    y_res = size(local_matrix,1)
    z_res = size(local_matrix,2)
    yfact = (-x1(2) + image_extent)/(2*image_extent)
    zfact = (x1(3) + image_extent)/(2*image_extent)
    if ((yfact > 0.0_rk) .and. (yfact < 1.0_rk) .and. (zfact > 0.0_rk) .and. (zfact < 1.0_rk)) then
        y_index = int(y_res*yfact) + 1
        z_index = int(z_res*zfact) + 1
        bmag = dsqrt(sum(k1**2))/maxb_overall
        altitude = dsqrt(sum(x1**2))/dexp(r(nr))
        longitude = x1(1)/altitude
        thomson_angle = abs(atan(sqrt(x1(2)**2 + x1(3)**2)/x1(1)))
        thomson_factor = sin(thomson_angle)**2
        overall_factor = thomson_factor*bmag**abs(image_parameters(1))
        !Determine magfield and location factors based on the field strenth etc.
        local_matrix(y_index, z_index) = local_matrix(y_index, z_index) + overall_factor
    end if

    end subroutine update_emissions

    !****************************************************************
    subroutine prepareInterpB(br, bs, bp, db, m, r, s, p, nr, ns, np, dr, ds, dp)
        integer,parameter :: rk=selected_real_kind(15,100)
        real(rk), intent(in), dimension(:,:,:) :: br, bs, bp
        real(rk), intent(in):: r(1:nr), s(1:ns), p(1:np)
        real(rk), dimension(:,:,:), allocatable :: brg, bsg, bpg
        real(rk), dimension(:,:,:,:,:):: db
        real(rk), dimension(64,64) :: m                          ! interpolation coefficients
        integer:: nr, ns, np

        ! ---
        ! Convert input vector field on cell faces to cartesian components at
        ! grid points. Then precompute derivatives and read in matrix for
        ! tricubic interpolation.
        ! - inputs: br(nr,ns+1,np+1), bs(nr+1,ns,np+1), bp(nr+1,ns+1,np)
        !             on cell faces
        ! - store in module-wide: db(8,nr,ns,np,3) -- bx, by, bz and derivatives, at grid pts.
        !                         m(64,64) -- interpolation matrix
        ! ---
        real(rk), dimension(:,:), allocatable :: atmp
        real(rk), dimension(:), allocatable :: rc, sc
        real(rk):: dr, ds, dp
        integer :: i,j

        ! Weighted average to grid points
        ! -------------------------------
        ! Coordinates at face centres:

        dr = sum(r(2:nr) - r(1:nr-1))/(nr-1)
        ds = sum(s(2:ns) - s(1:ns-1))/(ns-1)
        dp = sum(p(2:np) - p(1:np-1))/(np-1)

        allocate(rc(nr+1),sc(ns+1))
        rc = (/ (r(1) + (dble(i) - 0.5_rk)*dr, i=0,nr) /)
        sc = (/ (s(1) + (dble(i) - 0.5_rk)*ds, i=0,ns) /)

        if (.not.allocated(brg)) &
             allocate(brg(nr,ns,np),bsg(nr,ns,np),bpg(nr,ns,np))
        ! br (no need for weighting since cells have equal area)
        brg = 0.25_rk*(br(:,1:ns,1:np) + br(:,1:ns,2:np+1) &
            + br(:,2:ns+1,1:np) + br(:,2:ns+1,2:np+1))
        ! bs
        allocate(atmp(nr+1,ns))
        atmp = 0.0_rk
        do i=1,nr+1
        atmp(i,:) = 0.5_rk*(dexp(2.0_rk*dr) - 1.0_rk)*dp*dexp(2.0_rk*rc(i) - dr)&
        *dsqrt(abs(1.0_rk - s**2))
        end do
        atmp(:,1) = atmp(:,2)
        atmp(:,ns) = atmp(:,ns-1)
        do i=1,np
        bsg(:,:,i) = (bs(1:nr,:,i) + bs(1:nr,:,i+1))*atmp(1:nr,:) &
                + (bs(2:nr+1,:,i) + bs(2:nr+1,:,i+1))*atmp(2:nr+1,:)
        bsg(:,:,i) = 0.5*bsg(:,:,i)/(atmp(1:nr,:) + atmp(2:nr+1,:))
        end do

        deallocate(atmp)
        ! bp
        allocate(atmp(nr+1,ns+1))
        do i=1,nr+1
        atmp(i,2:ns) = 0.5_rk*(dexp(2.0_rk*dr) - 1.0_rk)*dexp(2.0_rk*rc(i) - dr) &
                *(dasin(s(2:ns)) - dasin(s(1:ns-1)))
        end do
        atmp(:,1) = atmp(:,2)
        atmp(:,ns+1) = atmp(:,ns)
        do i=1,np
        bpg(:,:,i) = bp(1:nr,1:ns,i)*atmp(1:nr,1:ns) &
                + bp(1:nr,2:ns+1,i)*atmp(1:nr,2:ns+1) &
                + bp(2:nr+1,1:ns,i)*atmp(2:nr+1,1:ns) &
                + bp(2:nr+1,2:ns+1,i)*atmp(2:nr+1,2:ns+1)
        bpg(:,:,i) = bpg(:,:,i)/(atmp(1:nr,1:ns) + atmp(1:nr,2:ns+1) &
                + atmp(2:nr+1,1:ns) + atmp(2:nr+1,2:ns+1))
        end do
        deallocate(atmp)

        ! Convert to cartesian cmpts at grid points, and estimate derivatives
        ! -------------------------------------------------------------------
        db = 0.0_rk
        ! Function itself:
        do i=1,np
            do j=1,ns
                db(1,:,j,i,1) = dsqrt(1.0_rk - s(j)**2)*cos(p(i))*brg(:,j,i) - s(j)*cos(p(i))*bsg(:,j,i) - sin(p(i))*bpg(:,j,i)
                db(1,:,j,i,2) = dsqrt(1.0_rk - s(j)**2)*sin(p(i))*brg(:,j,i) - s(j)*sin(p(i))*bsg(:,j,i) + cos(p(i))*bpg(:,j,i)
                db(1,:,j,i,3) = s(j)*brg(:,j,i) + dsqrt(1.0_rk - s(j)**2)*bsg(:,j,i)
            end do
        end do
        ! d/dr: (1-sided for boundaries)
        db(2,2:nr-1,:,:,:) = 0.5_rk*(db(1,3:nr,:,:,:) - db(1,1:nr-2,:,:,:))
        db(2,1,:,:,:) = 0.5_rk*(-3.0_rk*db(1,1,:,:,:) + 4.0_rk*db(1,2,:,:,:) - db(1,3,:,:,:))
        db(2,nr,:,:,:) = 0.5_rk*(3.0_rk*db(1,nr,:,:,:) - 4.0_rk*db(1,nr-1,:,:,:) + db(1,nr-2,:,:,:))
        ! d/ds: (1-sided for boundaries)
        db(3,:,2:ns-1,:,:) = 0.5_rk*(db(1,:,3:ns,:,:) - db(1,:,1:ns-2,:,:))
        db(3,:,1,:,:) = 0.5_rk*(-3.0_rk*db(1,:,1,:,:) + 4.0_rk*db(1,:,2,:,:) - db(1,:,3,:,:))
        db(3,:,ns,:,:) = 0.5_rk*(3.0_rk*db(1,:,ns,:,:) - 4.0_rk*db(1,:,ns-1,:,:) + db(1,:,ns-2,:,:))
        ! d/dp: (periodic for boundaries)
        db(4,:,:,2:np-1,:) = 0.5_rk*(db(1,:,:,3:np,:) - db(1,:,:,1:np-2,:))
        db(4,:,:,1,:) = 0.5_rk*(db(1,:,:,2,:) - db(1,:,:,np-1,:))
        db(4,:,:,np,:) = 0.5_rk*(db(1,:,:,2,:) - db(1,:,:,np-1,:))
        ! d^2/drds: (1-sided on boundaries)
        db(5,2:nr-1,:,:,:) = 0.5_rk*(db(3,3:nr,:,:,:) - db(3,1:nr-2,:,:,:))
        db(5,1,:,:,:) = 0.5_rk*(-3.0_rk*db(3,1,:,:,:) + 4.0_rk*db(3,2,:,:,:) - db(3,3,:,:,:))
        db(5,nr,:,:,:) = 0.5_rk*(3.0_rk*db(3,nr,:,:,:) - 4.0_rk*db(3,nr-1,:,:,:) + db(3,nr-2,:,:,:))
        ! d^2/drdp: (1-sided on boundaries)
        db(6,2:nr-1,:,:,:) = 0.5_rk*(db(4,3:nr,:,:,:) - db(4,1:nr-2,:,:,:))
        db(6,1,:,:,:) = 0.5_rk*(-3.0_rk*db(4,1,:,:,:) + 4.0_rk*db(4,2,:,:,:) - db(4,3,:,:,:))
        db(6,nr,:,:,:) = 0.5_rk*(3.0_rk*db(4,nr,:,:,:) - 4.0_rk*db(4,nr-1,:,:,:) + db(4,nr-2,:,:,:))
        ! d^2/dsdp: (1-sided on boundaries)
        db(7,:,2:ns-1,:,:) = 0.5_rk*(db(4,:,3:ns,:,:) - db(4,:,1:ns-2,:,:))
        db(7,:,1,:,:) = 0.5_rk*(-3.0_rk*db(4,:,1,:,:) + 4.0_rk*db(4,:,2,:,:) - db(4,:,3,:,:))
        db(7,:,ns,:,:) = 0.5_rk*(3.0_rk*db(4,:,ns,:,:) - 4.0_rk*db(4,:,ns-1,:,:) + db(4,:,ns-2,:,:))
        ! d^3/drdsdp: (1-sided on boundaries)
        db(8,2:nr-1,:,:,:) = 0.5_rk*(db(7,3:nr,:,:,:) - db(7,1:nr-2,:,:,:))
        db(8,1,:,:,:) = 0.5_rk*(-3.0_rk*db(7,1,:,:,:) + 4.0_rk*db(7,2,:,:,:) - db(7,3,:,:,:))
        db(8,nr,:,:,:) = 0.5_rk*(3.0_rk*db(7,nr,:,:,:) - 4.0_rk*db(7,nr-1,:,:,:) + db(7,nr-2,:,:,:))

        call interpMatrix(m)

    end subroutine prepareInterpB

    !****************************************************************
    subroutine interpB(x1, b1, db, m, r, s, p, nr, ns, np, dr, ds, dp)
        integer,parameter :: rk=selected_real_kind(15,100)
        real(rk), parameter :: TWOPI=8.0_rk*datan(1.0_rk)

        real(rk), dimension(3), intent(in) :: x1
        real(rk), dimension(3), intent(out) :: b1
        real(rk), intent(in), dimension(:,:,:,:,:) :: db
        real(rk), dimension(64,64) :: m
        real(rk), intent(in):: r(1:nr), s(1:ns), p(1:np)
        real(rk), intent(in):: dr, ds, dp
        integer, intent(in):: nr, ns, np
        ! ---
        ! C1 tricubic interpolation for vector on regular grid
        ! - using method of Lekien & Marsden 2005.
        ! ---
        real(rk) :: r1, s1, p1
        real(rk) :: fr, fs, fp
        integer :: ir, is, ip

        real(rk), dimension(3) :: al
        real(rk) :: fact
        integer :: i,j

        ! Identify required cell.
        ! -----------------------
        r1 = dsqrt(sum(x1*x1))
        s1 = x1(3)/r1
        p1 = mod(atan2(x1(2),x1(1)) + TWOPI, TWOPI)
        r1 = log(r1)
        if (r1 < r(1)) r1=r(1)

        ir = floor((r1-r(1))/dr) + 1
        is = floor(abs(s1 + 1.0_rk)/ds) + 1
        ip = floor(p1/dp) + 1

        fr = (r1 - r(ir))/dr
        fs = (s1 - s(is))/ds
        fp = (p1 - p(ip))/dp

        if (ir.ge.nr) then
            ir = ir-1
            fr = 1.0_rk
        end if
        if (is.eq.ns) then
            is = is-1
            fs = 1.0_rk
        end if

        ! Compute interpolation coefficients.
        ! -----------------------------------
        ! Loop through each combination of powers
        b1 = 0.0_rk
        do i = 1,64
            al = 0.0_rk
            do j = 0,7
                al = al + m(i,8*j+1)*db(j+1,ir,is,ip,:) &
                    + m(i,8*j+2)*db(j+1,ir+1,is,ip,:) &
                    + m(i,8*j+3)*db(j+1,ir,is+1,ip,:) &
                    + m(i,8*j+4)*db(j+1,ir+1,is+1,ip,:) &
                    + m(i,8*j+5)*db(j+1,ir,is,ip+1,:) &
                    + m(i,8*j+6)*db(j+1,ir+1,is,ip+1,:) &
                    + m(i,8*j+7)*db(j+1,ir,is+1,ip+1,:) &
                    + m(i,8*j+8)*db(j+1,ir+1,is+1,ip+1,:)
            end do
            fact = fr**mod(i-1,4)*fs**mod((i-1)/4,4)*fp**((i-1)/16)
            b1 = b1 + al*fact
        end do
    end subroutine interpB

    !****************************************************************
    subroutine interpMatrix(m)
        ! ---
        ! Initialize interpolation matrix m for tricubic interpolation.
        ! --
        integer,parameter :: rk=selected_real_kind(15,100)
        real(rk), dimension(64,64) :: m

        m(1, :) = (/1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(2, :) = (/0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(3, :) = (/-3, 3, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(4, :) = (/2, -2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(5, :) = (/0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(6, :) = (/0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(7, :) = (/0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, &
        3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, &
        -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(8, :) = (/0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, &
        -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, &
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(9, :) = (/-3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, &
        0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(10, :) = (/0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, &
        0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(11, :) = (/9, -9, -9, 9, 0, 0, 0, 0, 6, 3, -6, -3, 0, 0, 0, 0, 6, &
        -6, 3, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, &
        2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(12, :) = (/-6, 6, 6, -6, 0, 0, 0, 0, -3, -3, 3, 3, 0, 0, 0, 0, -4, &
        4, -2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, &
        -2, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(13, :) = (/2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, &
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(14, :) = (/0, 0, 0, 0, 0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, &
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(15, :) = (/-6, 6, 6, -6, 0, 0, 0, 0, -4, -2, 4, 2, 0, 0, 0, 0, -3, &
        3, -3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, &
        -1, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(16, :) = (/4, -4, -4, 4, 0, 0, 0, 0, 2, 2, -2, -2, 0, 0, 0, 0, 2, &
        -2, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, &
        1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(17, :) = (/0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(18, :) = (/0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(19, :) = (/0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(20, :) = (/0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(21, :) = (/0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(22, :) = (/0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0/)
        m(23, :) = (/0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, &
        3, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0/)
        m(24, :) = (/0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, &
        -2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0/)
        m(25, :) = (/0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, &
        0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(26, :) = (/0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0/)
        m(27, :) = (/0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 9, -9, -9, 9, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 6, 3, -6, -3, 0, 0, 0, 0, 6, &
        -6, 3, -3, 0, 0, 0, 0, 4, 2, 2, 1, 0, 0, 0, 0/)
        m(28, :) = (/0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, -6, 6, 6, -6, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, -3, -3, 3, 3, 0, 0, 0, 0, -4, &
        4, -2, 2, 0, 0, 0, 0, -2, -2, -1, -1, 0, 0, 0, 0/)
        m(29, :) = (/0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, &
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(30, :) = (/0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0/)
        m(31, :) = (/0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, -6, 6, 6, -6, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, -4, -2, 4, 2, 0, 0, 0, 0, -3, &
        3, -3, 3, 0, 0, 0, 0, -2, -1, -2, -1, 0, 0, 0, 0/)
        m(32, :) = (/0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 4, -4, -4, 4, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 2, 2, -2, -2, 0, 0, 0, 0, 2, &
        -2, 2, -2, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0/)
        m(33, :) = (/-3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, -1, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(34, :) = (/0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 0, 0, 3, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, -1, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(35, :) = (/9, -9, 0, 0, -9, 9, 0, 0, 6, 3, 0, 0, -6, -3, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 6, -6, 0, 0, 3, -3, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 4, 2, 0, 0, 2, 1, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(36, :) = (/-6, 6, 0, 0, 6, -6, 0, 0, -3, -3, 0, 0, 3, 3, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, -4, 4, 0, 0, -2, 2, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, -2, -2, 0, 0, -1, -1, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(37, :) = (/0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, &
        0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, &
        0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(38, :) = (/0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, &
        0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, -1, 0, 0, 0/)
        m(39, :) = (/0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, &
        -9, 0, 0, -9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, &
        3, 0, 0, -6, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, &
        -6, 0, 0, 3, -3, 0, 0, 4, 2, 0, 0, 2, 1, 0, 0/)
        m(40, :) = (/0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, &
        6, 0, 0, 6, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, &
        -3, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, &
        4, 0, 0, -2, 2, 0, 0, -2, -2, 0, 0, -1, -1, 0, 0/)
        m(41, :) = (/9, 0, -9, 0, -9, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, &
        0, 3, 0, -6, 0, -3, 0, 6, 0, -6, 0, 3, 0, -3, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, &
        0, 2, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(42, :) = (/0, 0, 0, 0, 0, 0, 0, 0, 9, 0, -9, 0, -9, 0, 9, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, &
        0, 3, 0, -6, 0, -3, 0, 6, 0, -6, 0, 3, 0, -3, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 4, 0, 2, 0, 2, 0, 1, 0/)
        m(43, :) = (/-27, 27, 27, -27, 27, -27, -27, 27, -18, -9, 18, 9, 18, 9, -18, -9, -18, &
        18, -9, 9, 18, -18, 9, -9, -18, 18, 18, -18, -9, 9, 9, -9, -12, &
        -6, -6, -3, 12, 6, 6, 3, -12, -6, 12, 6, -6, -3, 6, 3, -12, &
        12, -6, 6, -6, 6, -3, 3, -8, -4, -4, -2, -4, -2, -2, -1/)
        m(44, :) = (/18, -18, -18, 18, -18, 18, 18, -18, 9, 9, -9, -9, -9, -9, 9, 9, 12, &
        -12, 6, -6, -12, 12, -6, 6, 12, -12, -12, 12, 6, -6, -6, 6, 6, &
        6, 3, 3, -6, -6, -3, -3, 6, 6, -6, -6, 3, 3, -3, -3, 8, &
        -8, 4, -4, 4, -4, 2, -2, 4, 4, 2, 2, 2, 2, 1, 1/)
        m(45, :) = (/-6, 0, 6, 0, 6, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, &
        0, -3, 0, 3, 0, 3, 0, -4, 0, 4, 0, -2, 0, 2, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, &
        0, -2, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(46, :) = (/0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 6, 0, 6, 0, -6, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, &
        0, -3, 0, 3, 0, 3, 0, -4, 0, 4, 0, -2, 0, 2, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, -2, 0, -2, 0, -1, 0, -1, 0/)
        m(47, :) = (/18, -18, -18, 18, -18, 18, 18, -18, 12, 6, -12, -6, -12, -6, 12, 6, 9, &
        -9, 9, -9, -9, 9, -9, 9, 12, -12, -12, 12, 6, -6, -6, 6, 6, &
        3, 6, 3, -6, -3, -6, -3, 8, 4, -8, -4, 4, 2, -4, -2, 6, &
        -6, 6, -6, 3, -3, 3, -3, 4, 2, 4, 2, 2, 1, 2, 1/)
        m(48, :) = (/-12, 12, 12, -12, 12, -12, -12, 12, -6, -6, 6, 6, 6, 6, -6, -6, -6, &
        6, -6, 6, 6, -6, 6, -6, -8, 8, 8, -8, -4, 4, 4, -4, -3, &
        -3, -3, -3, 3, 3, 3, 3, -4, -4, 4, 4, -2, -2, 2, 2, -4, &
        4, -4, 4, -2, 2, -2, 2, -2, -2, -2, -2, -1, -1, -1, -1/)
        m(49, :) = (/2, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(50, :) = (/0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, -2, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(51, :) = (/-6, 6, 0, 0, 6, -6, 0, 0, -4, -2, 0, 0, 4, 2, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -3, 3, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, -2, -1, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(52, :) = (/4, -4, 0, 0, -4, 4, 0, 0, 2, 2, 0, 0, -2, -2, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 2, -2, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(53, :) = (/0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, &
        0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, &
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(54, :) = (/0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, &
        0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0/)
        m(55, :) = (/0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, &
        6, 0, 0, 6, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, &
        -2, 0, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, &
        3, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0, -2, -1, 0, 0/)
        m(56, :) = (/0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, &
        -4, 0, 0, -4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, &
        2, 0, 0, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, &
        -2, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0/)
        m(57, :) = (/-6, 0, 6, 0, 6, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, &
        0, -2, 0, 4, 0, 2, 0, -3, 0, 3, 0, -3, 0, 3, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, &
        0, -1, 0, -2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(58, :) = (/0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 6, 0, 6, 0, -6, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, &
        0, -2, 0, 4, 0, 2, 0, -3, 0, 3, 0, -3, 0, 3, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, -2, 0, -1, 0/)
        m(59, :) = (/18, -18, -18, 18, -18, 18, 18, -18, 12, 6, -12, -6, -12, -6, 12, 6, 12, &
        -12, 6, -6, -12, 12, -6, 6, 9, -9, -9, 9, 9, -9, -9, 9, 8, &
        4, 4, 2, -8, -4, -4, -2, 6, 3, -6, -3, 6, 3, -6, -3, 6, &
        -6, 3, -3, 6, -6, 3, -3, 4, 2, 2, 1, 4, 2, 2, 1/)
        m(60, :) = (/-12, 12, 12, -12, 12, -12, -12, 12, -6, -6, 6, 6, 6, 6, -6, -6, -8, &
        8, -4, 4, 8, -8, 4, -4, -6, 6, 6, -6, -6, 6, 6, -6, -4, &
        -4, -2, -2, 4, 4, 2, 2, -3, -3, 3, 3, -3, -3, 3, 3, -4, &
        4, -2, 2, -4, 4, -2, 2, -2, -2, -1, -1, -2, -2, -1, -1/)
        m(61, :) = (/4, 0, -4, 0, -4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, &
        0, 2, 0, -2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, &
        0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0/)
        m(62, :) = (/0, 0, 0, 0, 0, 0, 0, 0, 4, 0, -4, 0, -4, 0, 4, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, &
        0, 2, 0, -2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 0, &
        0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0/)
        m(63, :) = (/-12, 12, 12, -12, 12, -12, -12, 12, -8, -4, 8, 4, 8, 4, -8, -4, -6, &
        6, -6, 6, 6, -6, 6, -6, -6, 6, 6, -6, -6, 6, 6, -6, -4, &
        -2, -4, -2, 4, 2, 4, 2, -4, -2, 4, 2, -4, -2, 4, 2, -3, &
        3, -3, 3, -3, 3, -3, 3, -2, -1, -2, -1, -2, -1, -2, -1/)
        m(64, :) = (/8, -8, -8, 8, -8, 8, 8, -8, 4, 4, -4, -4, -4, -4, 4, 4, 4, &
        -4, 4, -4, -4, 4, -4, 4, 4, -4, -4, 4, 4, -4, -4, 4, 2, &
        2, 2, 2, -2, -2, -2, -2, 2, 2, -2, -2, 2, 2, -2, -2, 2, &
        -2, 2, -2, 2, -2, 2, -2, 1, 1, 1, 1, 1, 1, 1, 1/)

    end subroutine interpMatrix

end module fltrace
