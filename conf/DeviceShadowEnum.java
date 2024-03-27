package com.kejian.framework.energy.common.enumeration.delivery;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.apache.commons.lang3.StringUtils;

import cn.hutool.core.util.ObjectUtil;

/**
 * @author ：ronan
 * @date ：Created in 2023/4/26 13:37
 * @description：设备影子映射
 * @modified By：
 */
@Getter
@AllArgsConstructor
public enum DeviceShadowEnum {

    /**
     * 汽象温度T1
     */
    lf_T_R_VpTrT1("lf_T_R_VpTrT1", "汽相温度T1","T_R_VpTrT1", "TANK_TEAM_TEMP", "汽相温度T1 (℃)","esc_t_r_vptrt1", 1),
    /**
     * 液相温度T2
     */
    lf_T_R_LpTrT2("lf_T_R_LpTrT2", "液相温度T2","T_R_LpTrT2", "TANK_HOT_WATER_TEMP", "液相温度T2 (℃)","esc_t_r_lptrt2", 2),
    /**
     * 液相温度T3
     */
    lf_T_R_LpTrT3("lf_T_R_LpTrT3", "液相温度T3","T_R_LpTrT3", "TANK_REAR_COMPARTMENT_TEMP", "液相温度T3 (℃)","esc_t_r_lptrt3", 3),
    /**
     * 液位
     */
    lf_T_R_Ll("lf_T_R_Ll", "液位","T_R_Ll", "LIQUID_LEVEL", "液位 (mm)","esc_t_r_ll", 4),
    /**
     * 罐体压力P1
     */
    lf_T_R_TbPsP1("lf_T_R_TbPsP1", "罐体压力P1","T_R_TbPsP1", "TANK_PRESSURE", "罐体压力P1 (Mpa)","esc_t_r_tbpsp1", 5),
    /**
     * 供汽压力
     */
    lf_T_R_SsPs("lf_T_R_SsPs", "供汽压力","T_R_SsPs", "TANK_AIR_SUPPLY_PRESSURE", "供汽压力 (Mpa)","esc_t_r_ssps", 6),
    /**
     * 供水压力
     */
    lf_T_R_WsPs("lf_T_R_WsPs", "供水压力","T_R_WsPs", "TANK_WATER_SUPPLY_PRESSURE", "供水压力 (Mpa)","esc_t_r_wsps", 7),
    /**
     * 供汽开关阀开关位
     */
    lf_T_R_SsSvsb("lf_T_R_SsSvsb", "供汽开关阀开关位","T_R_SsSvsb", "TANK_VENT_SWITCH_VALVE", "供汽开关阀开关位","esc_t_r_sssvsb", 8),
    /**
     * 汽阀阀位
     */
    lf_T_R_Avvp("lf_T_R_Avvp", "汽阀阀位","T_R_Avvp", "TANK_VENT_REGULATING", "汽阀阀位","esc_t_r_avvp", 9),
    /**
     * 充汽开关阀开关位
     */
    lf_T_R_AiSvsb("lf_T_R_AiSvsb","充汽开关阀开关位","T_R_AiSvsb", "TANK_CHARGE_PORT_SWITCH_VALVE", "充汽开关阀开关位","esc_t_r_aisvsb", 10),
    /**
     * 供汽开关阀开位; 模拟转数字（供汽开关阀开关位）
     */
    lf_T_R_SsSvsbOn("lf_T_R_SsSvsbOn","供汽开关阀开位","T_R_SsSvsbOn", "", "","esc_t_t_sssvsbon", 11),
    /**
     * 供汽开关阀关位; 模拟转数字（供汽开关阀开关位）
     */
    lf_T_R_SsSvsbOff("lf_T_R_SsSvsbOff", "供汽开关阀关位","T_R_SsSvsbOff", "", "","esc_t_t_sssvsboff", 12),
    /**
     * 充汽开关阀开位; 模拟转数字（充汽开关阀开关位）
     */
    lf_T_R_AiSvsbOn("lf_T_R_AiSvsbOn", "充汽开关阀开位","T_R_AiSvsbOn", "", "","esc_t_r_aisvsbon", 13),
    /**
     * 充汽开关阀关位; 模拟转数字（充汽开关阀开关位）
     */
    lf_T_R_AiSvsbOff("lf_T_R_AiSvsbOff", "充汽开关阀关位","T_R_AiSvsbOff", "", "","esc_t_r_aisvsboff", 14),


    /**
     * 温度;  总流量计温度
     */
    lf_S_R_Tr("lf_S_R_Tr", "温度","S_R_Tr", "TEMP", "温度 (℃)","ces_s_r_tr", 15),
    /**
     * 压力; 总流量压力
     */
    lf_S_R_Ps("lf_S_R_Ps", "压力","S_R_Ps", "PRESSURE", "压力 (Kpa)","ces_s_r_ps", 16),
    /**
     * 瞬时; 瞬时流量; 工况流量; 总流量瞬时值
     */
    lf_S_R_Ifr("lf_S_R_Ifr", "瞬时流量","S_R_Ifr", "INSTANTANEOUS_FLOW", "瞬时流量 (t)","ces_s_r_ifr", 17),
    /**
     * 总流量; 累计流量; 总流量累计
     */
    lf_S_ST_Cd("lf_S_ST_Cd", "累计流量","S_ST_Cd", "CUMULATIVE_FLOW_UP", "总流量 (t)","ces_s_st_cd", 18),
    ;

    private final String name;

    private final String chinesName;

    private final String aliyunIotIdentifier;

    private final String lfIdentifier;

    private final String lfIdentifierName;

    private final String bigDataIdentifier;

    private final int signalId;


    public static DeviceShadowEnum getEnumByLfIdentifier(String lfIdentifier){
        if(StringUtils.isEmpty(lfIdentifier)){
            return null;
        }
        for (DeviceShadowEnum deviceShadowEnum : DeviceShadowEnum.values()) {
            if(deviceShadowEnum.getLfIdentifier().equals(lfIdentifier)){
                return deviceShadowEnum;
            }
        }
        return null;
    }

    /**
     * 根据信号ID获取枚举
     *
     * @param signId
     * @return
     */
    public static DeviceShadowEnum getEnumBySignalId(Integer signId){
        if(ObjectUtil.isNull(signId)){
            return null;
        }
        for (DeviceShadowEnum deviceShadowEnum : DeviceShadowEnum.values()) {
            if(deviceShadowEnum.getSignalId() == signId){
                return deviceShadowEnum;
            }
        }
        return null;
    }

}
