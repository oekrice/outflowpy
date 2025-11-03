module compute_outflow
    implicit none
    contains

    subroutine compute_outeqm(brsurf, rg, ss, p, rcx, sc, vc_in, dv_in, ms, ls, trigs, legs, nr, ns, np, br, bs, bp)
!     !   v1 is the speed parameter (in km/s), but this is irrelevant as python will do that calculation to allow for more complex inputs than Fortran would allow with ease
!     ! ---
    integer,parameter :: rk=selected_real_kind(15,100)
    integer, intent(in):: nr, ns, np
    real(rk), intent(in):: brsurf(0:ns-1,0:np-1)
    real(rk), intent(in):: rg(0:nr), ss(0:ns), p(0:np), rcx(-1:nr), sc(0:ns-1)
    real(rk), intent(in):: vc_in(-1:nr), dv_in(-1:nr)
    real(rk), intent(out):: br(0:nr  , 0:ns-1, 0:np-1), bs(0:nr-1, 0:ns  , 0:np-1), bp(0:nr-1, 0:ns-1, 0:np  )

    !Then all the various dummy variables
    real(rk):: dp, ds, dr
    real(rk), intent(in):: ms(0:np-1), trigs(0:np-1,0:np-1)
    real(rk), intent(in):: ls(0:np-1,0:ns-1), legs(0:np-1,0:ns-1,0:ns-1)
    real(rk):: sig(0:ns),  sigc(0:ns-1)
    real(rk):: cml, grad
    real(rk):: vc(-1:nr), dv(-1:nr)
    integer:: l, m, i, j, k
    real(rk):: A, B, C, top, bottom
    real(rk):: hc(-1:nr,0:np-1,0:ns-1), g(0:nr,0:np-1,0:ns-1)

    print*, 'Computing outflow equilibrium... '

    vc = vc_in; dv = dv_in
    br = 0.0; bs = 1.0; bp = 1.0
    ! Compute azimuthal eigenvalues m and eigenvectors:
    dp = sum((p(1:np) - p(0:np-1))/np)

    !call findms(ms, trigs, dp, np)
    ! Find latitudinal eigenvalues l and eigenvectors, for each m:

    ds = sum((ss(1:ns) - ss(0:ns-1))/ns)

    dr = sum(rg(1:nr) - rg(0:nr-1))/nr

!     do m=0, np-1
!          call findls(ms(m), ls(m,:), legs(m,:,:), ds , ss, sc, ns)
!     end do


    print*, 'Eigenthings calculated.'

    sig = (1.0_rk - ss**2)**0.5_rk
    sigc = (1.0_rk - sc**2)**0.5_rk

    !I've commented these out as I don't think they're appropriate here! I might be wrong -- worth checking with Anthony?
    !vc = vc*radc !Hmm. Why?!
    !dv = dv*radc

    ! Compute c_ml*[radial functions] for each m and l:
    hc = 0.0_rk
    g = 0.0_rk

    !This is all the non-radial stuff. So could put proper checks in here if necessary
    do m=0,np-1
         do l=0,ns-1
            ! Compute coefficient cml using orthogonality of eigenvectors:
            cml = 0.0_rk
            B = 0.0_rk  ! denominator
            do k=0,np-1
                do j=0,ns-1
                    A = trigs(k,m)*legs(m,j,l)
                    cml = cml + A*brsurf(j,k)
                    B = B + A**2
                end do
            end do

            !B should always be 1, and it is (pretty much!)
            cml = cml/B

            if (abs(cml) > 1d-10) then

            hc(nr,m,l) = 1.0_rk  ! This numerical option matches PFSSpy
            hc(nr-1,m,l) = 0.0_rk

            g(0,m,l) = 1.0_rk   ! lower boundary condition

            ! Find radial function H_l(rho) at cell centres
            ! - integrate backwards from the outer boundary:
            do i=nr-2, -1, -1
                A = 1.0_rk
                B = 3.0_rk - vc(i+1)*dexp(rcx(i+1))
                C = 2.0_rk - ls(m,l) - 3.0_rk*vc(i+1)*dexp(rcx(i+1)) - dv(i+1)*dexp(rcx(i+1))
                top = hc(i+1, m, l) * (2*A - C*dr**2) + hc(i+2,m,l)*(-A - 0.5_rk*dr*B)
                bottom = (A - 0.5_rk*dr*B)
                hc(i, m, l) = top/bottom
            end do

            ! - normalize to satisfy lower boundary condition:
            !hc(:,m,l) = hc(:,m,l)/((hc(0,m,l) - hc(-1,m,l))/dr + (1-vc(-1)*dexp(r(0)))*hc(-1,m,l))  ! modified to allow v non-zero on lower boundary
            !hc(:,m,l) = hc(:,m,l)/((hc(0,m,l) - hc(-1,m,l))/dr + (1-0.5*(vc(-1) + vc(0))*dexp(r(0)))*0.5*(hc(-1,m,l) + hc(0,m,l)))  ! version 1
            !hc(:,m,l) = hc(:,m,l)/((hc(0,m,l) - hc(-1,m,l))/dr + (1-0.5*(vc(-1) + vc(0))*dexp(r(0)))*hc(-1,m,l))  ! version 2
            !hc(:,m,l) = hc(:,m,l)/((hc(0,m,l) - hc(-1,m,l))/dr + (1-vc(-1)*dexp(rcx(-1)))*hc(-1,m,l))  ! version 3. THIS APPEARS TO BE THE MOST REALISTIC FOR SOME REASON. ANTHONY SAID UPWINDING?

            grad = (hc(0,m,l)*dexp(rcx(0)) - hc(-1,m,l)*dexp(rcx(-1)))/dr - vc(-1)*dexp(rcx(-1))*hc(-1,m,l)


            !Allowance here for difference in the position of hc and rc measurement
            hc(:,m,l) = hc(:,m,l)/grad

            ! Find G_l(rho) from H_l(rho) using recurrence.
            do i=1,nr
                g(i,m,l) = ( 0.5_rk*ls(m,l)*hc(i-1,m,l)*(dexp(2*rg(i)) - dexp(2*rg(i-1))) &
                + g(i-1,m,l)*dexp(2*rg(i-1)) ) / dexp(2*rg(i))
            end do

            hc(:,m,l) = hc(:,m,l)*cml  ! for later convenience
            g(:,m,l) = g(:,m,l)*cml
            end if

         end do
    end do

    print*, 'Radial functions calculated.'

    br = 0.0_rk; bs = 0.0_rk; bp = 0.0_rk
    ! Fill up magnetic field components for each m:
    do j=0,ns-1
        do i=0,nr
            do m=0,np-1
                A = sum(g(i,m,:)*legs(m,j,:))
                do k=0,np-1
                    br(i,j,k) = br(i,j,k) + A*trigs(k,m)
                end do
            end do
        end do
    end do


    print*, 'Max lower boundary error:', maxval(br(0,:,:) - brsurf)
    do j=1,ns-1  !Don't need to do the poles as this will have zero area anyway -- not problematic
        do i=0,nr-1
            do m=0,np-1
                A = sig(j)*sum(hc(i,m,:)*(legs(m,j,:) - legs(m,j-1,:)))/ds
                do k=0,np-1
                    bs(i,j,k) = bs(i,j,k) + A*trigs(k,m)
                end do
            end do
        end do
    end do

    do j=0,ns-1
        do i=0,nr-1
            do m=1,np-1   !This misses the ends -- will need to duplicate them somehow. The python code manages it...
                A = sum(hc(i,m,:)*legs(m,j,:))/sigc(j)/dp
                do k=1,np-1
                    bp(i,j,k) = bp(i,j,k) + A*(trigs(k,m) - trigs(k-1,m))
                end do
                bp(i,j,0) = bp(i,j,0) + A*(trigs(0,m) - trigs(np-1,m))
            end do
            bp(i,j,np) = bp(i,j,0)
        end do
    end do

    print*, 'Magnetic field calculated. Lovely.'
!     print*, 'fortran br sum', sum(abs(br(:,:,:)))
!     print*, 'fortran bs sum', sum(abs(bs(:,:,:)))
!     print*, 'fortran bp sum', sum(abs(bp(:,:,:)))


    end subroutine compute_outeqm


end module compute_outflow
