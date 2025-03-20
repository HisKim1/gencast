;-----------------------------------------------------------
; Script: regrid_NE120_to_N360.ncl
; 목적: CESM2 ne120_t12 grid의 U 변수를 N360 Gaussian grid로 보간
; 참고: ECMWF 2.5도 grid 보간 예제와 unstructured -> FV grid 보간 예제를 참조
;-----------------------------------------------------------

; 필요한 라이브러리 로드
load "$NCARG_ROOT/lib/ncarg/nclscripts/esmf/ESMF_regridding.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"

;-----------------------------------------------------------
; 사용자 설정
;-----------------------------------------------------------
remap_method = "bilinear"       ; 보간 방법 (예: "bilinear", "patch", "conserve")
SE_res       = "ne120_t12"        ; CESM2 SE grid 해상도
DstGridName  = "N360"             ; 목적 그리드: N360 Gaussian

; 가중치 파일 생성 디렉토리 및 파일명
WgtFileDir  = "./"
WgtFileName = "SE_" + str_upper(SE_res) + "_to_" + DstGridName + "." + remap_method + ".nc"

; 입력 파일: CESM2 데이터 (U.nc)
inFile = "/geodata2/Gencast/CESM2/hires/U.nc"
sfile  = addfile(inFile, "r")

; 목적 그리드 파일 (N360 Gaussian grid 정의 파일)
; 이 파일에는 변수 "lat"와 "lon"이 있어야 함.
dstGridFile = "/geodata2/Gencast/CESM2/hires/n360.nc"
dfile  = addfile(dstGridFile, "r")

;-----------------------------------------------------------
; 원본 변수 및 그리드 정보 읽기
;-----------------------------------------------------------
var_name = "U"             ; 보간할 변수 이름
var_in   = sfile->$var_name$
src_lat  = sfile->lat      ; 원본 위도 (ncol 차원)
src_lon  = sfile->lon      ; 원본 경도 (ncol 차원)

; 목적 그리드 위도/경도 읽기 (N360 Gaussian grid)
dst_lat = dfile->lat
dst_lon = dfile->lon

;-----------------------------------------------------------
; ESMF 재그리드 옵션 설정
;-----------------------------------------------------------
Opt = True
Opt@InterpMethod      = remap_method
Opt@WgtFileName       = WgtFileDir + WgtFileName
Opt@SrcGridLat        = src_lat
Opt@SrcGridLon        = src_lon
Opt@SrcInputFileName  = inFile
Opt@DstGridLat        = dst_lat
Opt@DstGridLon        = dst_lon
Opt@ForceOverwrite    = True
Opt@Debug             = True
Opt@PrintTimings      = True
; ElementVertices 옵션은 제공하지 않음 (자동 계산에 맡김)
; (Opt@ElementVertices는 설정하지 않음)

Opt@SrcGridType = "unstructured"
Opt@DstGridType = "G360"
;-----------------------------------------------------------
; ESMF를 이용한 보간 수행
;-----------------------------------------------------------
print("Regridding " + var_name + " from " + SE_res + " grid to " + DstGridName + " grid using " + remap_method + "interpolation...")
var_regrid = ESMF_regrid(var_in, Opt)
printVarSummary(var_regrid)

;-----------------------------------------------------------
; (선택사항) 결과 확인용 플롯 작성
;-----------------------------------------------------------
wks = gsn_open_wks("png", "U_regrid")
res = True
res@gsnMaximize = True
res@gsnAddCyclic = True

; 원본 데이터 (첫 시간, 만약 time 차원이 있을 경우)
plot_orig = gsn_csm_contour_map(wks, var_in(0, :, :), res)
; 보간된 데이터 (첫 시간)
plot_regrid = gsn_csm_contour_map(wks, var_regrid(0, :, :), res)

; 두 플롯을 한 페이지에 패널로 출력
gsn_panel(wks, (/plot_orig, plot_regrid/), (/2, 1/), True)

;-----------------------------------------------------------
; 보간된 데이터를 새로운 NetCDF 파일로 저장
;-----------------------------------------------------------
outFile = "./U_N360_" + remap_method + ".nc"
system("/bin/rm -f " + outFile)
fout = addfile(outFile, "c")

; 목적 그리드 차원 크기 설정
nlat  = dimsizes(dst_lat)
nlon  = dimsizes(dst_lon)
ntime = dimsizes(var_regrid(:,0,0))   ; 시간 차원이 있을 경우

; define mode 진입
setfileoption(fout, "DefineMode", True)

; 전역 속성 정의
fAtt = True
fAtt@title = "Regridded " + var_name + " from " + SE_res + " grid to " + DstGridName + " grid"
fileattdef(fout, fAtt)
copy_VarAtts(sfile, fout)

; 차원 정의
dimNames = (/"time", "lat", "lon"/)
dimSizes = (/ ntime, nlat, nlon /)
dimUnlim = (/ True, False, False /)
filedimdef(fout, dimNames, dimSizes, dimUnlim)

; 변수 정의
filevardef(fout, "time", typeof(var_in(0,0,0)), "time")
filevardef(fout, "lat", typeof(dst_lat), "lat")
filevardef(fout, "lon", typeof(dst_lon), "lon")
filevardef(fout, var_name, typeof(var_regrid(0,0,0)), (/"time", "lat", "lon"/))

; define mode 종료
setfileoption(fout, "DefineMode", False)

; 변수 기록 (여기서는 시간축을 0,1,...로 생성)
fout->time = fspan(0, ntime - 1, ntime)
fout->lat  = dst_lat
fout->lon  = dst_lon
fout->$var_name$ = var_regrid

print("Regridded data written to " + outFile)

end
